#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <direct.h>
#include <iostream>
#include <windows.h>
#include <string>
#include <vector>
#include <stack>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, the jittering you can see below in jitter_image() was used during
// training, and the training dataset consisted of about 3 million images instead of 55.
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
	const matrix<rgb_pixel>& img
);

bool ListFiles(wstring path, wstring mask, std::vector<wstring>& files);

// ----------------------------------------------------------------------------------------
struct info
{
	int count;
	string name;
};

// ----------------------------------------------------------------------------------------
int main(int argc, char** argv) try
{

	frontal_face_detector detector = get_frontal_face_detector();

	shape_predictor sp;
	wchar_t buffer[MAX_PATH];
	GetModuleFileName(NULL, buffer, MAX_PATH);
	wstring ws(buffer);
	string path(ws.begin(), ws.end());
	path = path.substr(0, path.find_last_of("\\/") + 1);
	deserialize(path + "shape_predictor_68_face_landmarks.dat") >> sp;

	anet_type net;
	deserialize(path + "dlib_face_recognition_resnet_model_v1.dat") >> net;

	std::vector<wstring> files;
	const size_t size = strlen(argv[1]) + 1;
	wchar_t* path_source = new wchar_t[size];
	mbstowcs(path_source, argv[1], size);

	std::vector<matrix<rgb_pixel>> faces;
	std::vector<string> face_names;
	//std::vector<image_window> win_detect(10);
	int count = 0;
	if (ListFiles(path_source, L"*", files))
	{
		for (std::vector<wstring>::iterator it = files.begin();it != files.end(); ++it)
		{
			matrix<rgb_pixel> img;
			string path_file_img(it->begin(), it->end());
			load_image(img, path_file_img);

			cout << "Load : " << path_file_img << endl;

			string fName(path_file_img);
			fName = fName.substr(0, fName.find_last_of("\\/"));
			fName = fName.substr(fName.find_last_of("\\/"));
			/*win_detect[count].set_title("face cluster " + cast_to_string(count));
			win_detect[count].set_image(img);*/

			for (auto face : detector(img))
			{
				auto shape = sp(img, face);
				matrix<rgb_pixel> face_chip;
				extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
				faces.push_back(move(face_chip));
				face_names.push_back(fName);
				//win_detect[count].add_overlay(face);
			}
			count++;
		}
	}

	if (faces.size() == 0)
	{
		cout << "No faces found in image!" << endl;
		return 1;
	}
	std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);

	std::vector<sample_pair> edges;
	for (size_t i = 0; i < face_descriptors.size(); ++i)
	{
		for (size_t j = i + 1; j < face_descriptors.size(); ++j)
		{
			if (length(face_descriptors[i] - face_descriptors[j]) < (float)atof(argv[2]))
				edges.push_back(sample_pair(i, j));
		}
	}
	std::vector<unsigned long> labels;
	const auto num_clusters = chinese_whispers(edges, labels);
	cout << "number of people found in the image: " << num_clusters << endl;

	//std::vector<image_window> win_clusters(num_clusters);
	array2d<rgb_pixel> img2d;
	for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
	{
		std::vector<matrix<rgb_pixel>> temp;
		string path_save(path + cast_to_string(cluster_id));
		mkdir(path_save.c_str());

		for (size_t j = 0; j < labels.size(); ++j)
		{
			if (cluster_id == labels[j])
			{
				temp.push_back(faces[j]);
				assign_image(img2d, temp.back());
				save_jpeg(img2d, path_save + "/" + face_names[j] + cast_to_string(j) + ".jpg");
			}

		}
		/*win_clusters[cluster_id].set_title("face cluster " + cast_to_string(cluster_id));
		win_clusters[cluster_id].set_image(tile_images(temp));*/
	}

	cout << "success" << endl;

	

	/*cout << "face descriptor for one face: " << trans(face_descriptors[0]) << endl;

	matrix<float, 0, 1> face_descriptor = mean(mat(net(jitter_image(faces[0]))));
	cout << "jittered face descriptor for one face: " << trans(face_descriptor) << endl;*/

	cout << "hit enter to terminate" << endl;
	cin.get();
}
catch (std::exception& e)
{
	cout << e.what() << endl;
}

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
	const matrix<rgb_pixel>& img
)
{
	thread_local random_cropper cropper;
	cropper.set_chip_dims(150, 150);
	cropper.set_randomly_flip(true);
	cropper.set_max_object_height(0.99999);
	cropper.set_background_crops_fraction(0);
	cropper.set_min_object_height(0.97);
	cropper.set_translate_amount(0.02);
	cropper.set_max_rotation_degrees(3);

	std::vector<mmod_rect> raw_boxes(1), ignored_crop_boxes;
	raw_boxes[0] = shrink_rect(get_rect(img), 3);
	std::vector<matrix<rgb_pixel>> crops;

	matrix<rgb_pixel> temp;
	for (int i = 0; i < 100; ++i)
	{
		cropper(img, raw_boxes, temp, ignored_crop_boxes);
		crops.push_back(move(temp));
	}
	return crops;
}

bool ListFiles(wstring path, wstring mask, std::vector<wstring>& files) {
	HANDLE hFind = INVALID_HANDLE_VALUE;
	WIN32_FIND_DATA ffd;
	wstring spec;
	stack<wstring> directories;

	directories.push(path);
	files.clear();

	while (!directories.empty()) {
		path = directories.top();
		spec = path + L"/" + mask;
		directories.pop();

		hFind = FindFirstFile(spec.c_str(), &ffd);
		if (hFind == INVALID_HANDLE_VALUE) {
			return false;
		}

		do {
			if (wcscmp(ffd.cFileName, L".") != 0 &&
				wcscmp(ffd.cFileName, L"..") != 0) {
				if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
					directories.push(path + L"/" + ffd.cFileName);
				}
				else {
					files.push_back(path + L"/" + ffd.cFileName);
				}
			}
		} while (FindNextFile(hFind, &ffd) != 0);

		if (GetLastError() != ERROR_NO_MORE_FILES) {
			FindClose(hFind);
			return false;
		}

		FindClose(hFind);
		hFind = INVALID_HANDLE_VALUE;
	}

	return true;
}
