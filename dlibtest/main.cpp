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

bool ListFiles(wstring path, wstring mask, std::vector<wstring>& files);

bool is_file_exist(const char *fileName);

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
	
	int count = 0;
	if (ListFiles(path_source, L"*", files))
	{
		for (std::vector<wstring>::iterator it = files.begin();it != files.end(); ++it)
		{
			matrix<rgb_pixel> img;
			string path_file_img(it->begin(), it->end());
			load_image(img, path_file_img);

			cout << "Load : " << path_file_img << endl;

			/*string fName(path_file_img);
			fName = fName.substr(0, fName.find_last_of("\\/"));
			fName = fName.substr(fName.find_last_of("\\/"));*/

			for (auto face : detector(img))
			{
				auto shape = sp(img, face);
				matrix<rgb_pixel> face_chip;
				extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
				faces.push_back(move(face_chip));
				//face_names.push_back(fName);
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
	/*std::vector<matrix<float, 0, 1>> tmp = net(faces);

	//Check descriptors file for raeding and writing
	string path_des = path + "des.dat";
	size_t begin = 0;
	if (is_file_exist(path_des.c_str()))
	{
		deserialize(path_des) >> face_descriptors;
		begin = face_descriptors.size();
		face_descriptors.insert(face_descriptors.end(), tmp.begin(), tmp.end());
	}
	else
		face_descriptors = tmp;

	serialize(path_des) << face_descriptors;*/

	//Prepare data for graphing
	std::vector<sample_pair> edges;
	for (size_t i = 0; i < face_descriptors.size(); ++i)
	{
		for (size_t j = i + 1; j < face_descriptors.size(); ++j)
		{
			if (length(face_descriptors[i] - face_descriptors[j]) < (float)atof(argv[2]))
				edges.push_back(sample_pair(i, j));
		}
	}

	//Clustering
	std::vector<unsigned long> labels;
	const auto num_clusters = chinese_whispers(edges, labels, 200);
	cout << "number of people found in the image: " << num_clusters << endl;

	//Check label and group them
	array2d<rgb_pixel> img2d;
	std::vector<int> number_sf;
	for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
	{
		/*std::vector<matrix<rgb_pixel>> temp;
		string path_save(path + cast_to_string(cluster_id));
		mkdir(path_save.c_str());*/

		number_sf.push_back(0);

		for (size_t j = 0 ; j < labels.size(); ++j)
		{
			if (cluster_id == labels[j])
			{
				number_sf[cluster_id]++;
				/*temp.push_back(faces[j - begin]);
				assign_image(img2d, temp.back());
				save_jpeg(img2d, path_save + "/" + face_names[j - begin] + "_" + cast_to_string(j) + ".jpg");*/
			}

		}
	}

	//Clean data and save description
	int index, max;

	for (int i = 0; i < num_clusters; i++)
	{
		if (number_sf[i] > max)
		{
			max = number_sf[i];
			index = i;
		}
	}

	std::vector<matrix<rgb_pixel>> temp;
	std::vector<matrix<float, 0, 1>> face_each_descriptors;

	string fName(argv[1]);
	fName = fName.substr(fName.find_last_of("\\/"));
	string path_save(path + "description/");
	if (!is_file_exist(path_save.c_str()))
		mkdir(path_save.c_str());

	for (int i = 0; i < labels.size(); i++)
	{
		if (labels[i] == index)
		{
			face_each_descriptors.push_back(face_descriptors[i]);
			/*temp.push_back(faces[i]);
			assign_image(img2d, temp.back());
			save_jpeg(img2d, path_save + "_" + cast_to_string(i) + ".jpg");*/
		}
	}
	
	//Save description file	
	serialize(path_save + fName + ".dat") << face_each_descriptors;
	
	cout << "success" << endl;

}
catch (std::exception& e)
{
	cout << e.what() << endl;
}

// ----------------------------------------------------------------------------------------

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

bool is_file_exist(const char *fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}
