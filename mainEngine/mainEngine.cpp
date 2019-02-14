#include "mainEngine.h"

MainEngine::MainEngine(int dataset_id)
{
	random_colors.resize(10000);
	for (int label = 0; label < 10000; ++label)
	{
		random_colors[label] = cv::Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
	}

	if (dataset_id == 0)
	{
		rgbdEngine = new RGBDSceneReader();
		dataEngine = static_cast<DatasetReader*>(rgbdEngine);
		dataset_name = "rgbd";
	}
	else if(dataset_id == 1)
	{
		nyuv2Engine = new NYUv2Reader();
		dataEngine = static_cast<DatasetReader*>(nyuv2Engine);
		dataset_name = "nyuv2";
	}
	else if (dataset_id == 2)
	{
		cofEngine = new CoFusionReader();
		dataEngine = static_cast<DatasetReader*>(cofEngine);
		dataset_name = "cofusion";
	}
	else{
		std::cerr << "dataset Number invalid..." << std::endl;
		return;
	}


	width = dataEngine->getWidth();
	height = dataEngine->getHeight();
	sceneNum = dataEngine->getSceneNum();
	std::cout << "Load " << dataset_name << " RGBD Scene dataset" << std::endl;
	convEngine = new DimensionConvertor();
	convEngine->setCameraParameters(dataEngine->getIntrinsics(), width, height);
	normEngine = new NormalMapGenerator(width, height);

	filtEngine = new JointBilateralFilter(width, height);
	segmEngine = new GeometricSegmentation(width, height, random_colors);
	detectEngine = new ObjectDetector(random_colors);

	//photoSegmEngine = new photometricSegmentation();
	//photoSegmEngineMT = new photometricSegmentationMT();

	//shared_rgbImage = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
	//thsegm = new thread(&photometricSegmentationMT::startThread, photoSegmEngineMT, shared_rgbImage);

	CUDA_SAFE_CALL( cudaMallocHost((void **)&inputDepth_Host, sizeof(float)*width*height));
	CUDA_SAFE_CALL( cudaMalloc((void **)&points3D_Device, sizeof(float3)*width*height));

	CUDA_SAFE_CALL( cudaMalloc((void **)&normalMap_Device, sizeof(float3)*width*height));
	geoSegMap_Host = new int[width * height];
	objSegMap_Host = new int[width * height];
	cout << "Initialized ...." << endl;
}

MainEngine::~MainEngine()
{

}

bool areaCompare(bbox_t bbox1, bbox_t bbox2) { return (bbox1.w * bbox1.h) < (bbox2.w * bbox2.h); }


void MainEngine::assignClass2GeoSegBB(vector<bbox_t> detect_bbox, vector<bbox_t>& geoseg_bbox)
{
	sort(detect_bbox.begin(), detect_bbox.end(), areaCompare);
	for (size_t i = 0; i < detect_bbox.size(); i++)
	{
		bbox_t base_box = detect_bbox[i];
		float g_xmin = static_cast<float>(detect_bbox[i].x                   );
		float g_ymin = static_cast<float>(detect_bbox[i].y                   );
		float g_xmax = static_cast<float>(detect_bbox[i].x + detect_bbox[i].w);
		float g_ymax = static_cast<float>(detect_bbox[i].y + detect_bbox[i].h);

		float base_box_w = g_xmax - g_xmin;
		float base_box_h = g_ymax - g_ymin;
		for (size_t b = 0; b < geoseg_bbox.size(); b++)
		{
			bbox_t compare_box = geoseg_bbox[b];
			
			//ignore bounding box whose object label is already assigned.
			if (compare_box.obj_id != -1)
				continue;
			
			float c_xmin = static_cast<float>(compare_box.x);
			float c_ymin = static_cast<float>(compare_box.y);
			float c_xmax = static_cast<float>(compare_box.x + compare_box.w);
			float c_ymax = static_cast<float>(compare_box.y + compare_box.h);

			float xmin = std::max(c_xmin, g_xmin);
			float ymin = std::max(c_ymin, g_ymin);
			float xmax = std::min(c_xmax, g_xmax);
			float ymax = std::min(c_ymax, g_ymax);

			float w = static_cast<float>(std::max(0.0, double(xmax - xmin)));
			float h = static_cast<float>(std::max(0.0, double(ymax - ymin)));

			if (w * h < 0.0)
				continue;
			float test_box_w = c_xmax - c_xmin;
			float test_box_h = c_ymax - c_ymin;
			float inter_ = w * h;
			float union_ = test_box_h * test_box_w + base_box_h * base_box_w - inter_;
			
			float iou = inter_ / (test_box_w * test_box_h);

			if (iou > IoU_THRESH)
				geoseg_bbox[b].obj_id = base_box.obj_id;

		}
	}
}

void MainEngine::assignClass2GeoSegMap(vector<bbox_t> geoseg_bbox, int* objSegMap_Host)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int j = 0; j < width*height; j++)
	{
		int geoLabel = geoSegMap_Host[j];
		if (geoLabel == -1)
			continue;
		objSegMap_Host[j] = geoseg_bbox[geoLabel].obj_id;
	}
}

void MainEngine::geoSeg()
{
	filtEngine->Process(inputDepth_Host);	
	convEngine->projectiveToReal(filtEngine->getFiltered_Device(), points3D_Device);
	normEngine->generateNormalMap(points3D_Device);
	normalMap_Device = normEngine->getNormalMap();
	segmEngine->CalcEdge(points3D_Device, normalMap_Device);
	//segmEngine->CalcEdge_vis(points3D_Device, normalMap_Device);

	//cv::Mat segmMap = segmEngine->getSegmImg();
	//cv::imshow("segmMap", segmMap);
	memset(geoSegMap_Host, 1, sizeof(int) * width * height);
	
	segmEngine->labeling(true, geoSegMap_Host);
	segmEngine->labeling(false, geoSegMap_Host);
}


void MainEngine::Activate()
{


	for (int sidx = 1; sidx < sceneNum; sidx++)
	{
		int frameNum = dataEngine->getFrameNum(sidx);

		//VideoWriter writer("output_"  + to_string(sidx) + ".avi", CV_FOURCC('D', 'I', 'V', 'X'), 15, cv::Size(width * 2, height * 2));
		cout << "frame Max : " << frameNum << endl;		
		for (int fidx = 1; fidx < frameNum; fidx++)
		{
			size_t free_mem, total_mem;
  			cudaError_t cuda_status=cudaMemGetInfo(&free_mem, &total_mem);
  			std::cout << cuda_status << "," << free_mem/1024/1024 << "," << total_mem/1024/1024 << std::endl;
			clock_t start = clock();
			Mat rgb = dataEngine->getRGBImg(sidx, fidx);
			Mat depth = dataEngine->getDepthImg(sidx, fidx);
			cv::Mat visRgb = rgb.clone();
			cv::Mat visDepth = visualizeFloatImage(depth, true, false);
			inputDepth_Host = (float*)depth.data;
			thread detect_thread(&ObjectDetector::detect, detectEngine, rgb);
			//thread geoseg_thread(&MainEngine::geoSeg, this);

			//photoSegmEngine->segment(rgb);
			//cv::Mat photoSegmImg = photoSegmEngine->getsegmImg();
			//cv::Mat bin_photoSegm;
			//cv::adaptiveThreshold(photoSegmImg, bin_photoSegm, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
			//cv::imshow("bin_photosegm", bin_photoSegm);
			//cv::Mat photovis;
			//cv::cvtColor(photoSegmImg,photovis, CV_GRAY2BGR);

			geoSeg();
			
			//geoseg_thread.join();

			//cv::Mat phiMap = visualizeFloatImage(segmEngine->getPhiImg(), false, false);
			//cv::Mat kaiMap = visualizeFloatImage(segmEngine->getKaiImg(), false, false);

			cv::Mat labelImg = segmEngine->getVisualizedLabelImg(geoSegMap_Host);
			cv::imshow("label", labelImg);
			vector<bbox_t> geoBB = segmEngine->getGeometricBBox();

			detect_thread.join();
			vector<bbox_t> bbox = detectEngine->getBB();
			cv::Mat detectRes = detectEngine->visBB(rgb);

			assignClass2GeoSegBB(bbox, geoBB);
			memset(objSegMap_Host, -1, sizeof(int) * width * height);
			assignClass2GeoSegMap(geoBB, objSegMap_Host);

			cv::Mat classLabelImg = segmEngine->getVisualizedLabelImg(objSegMap_Host);
			cv::imshow("classLabel", classLabelImg);

			cv::Mat mask = classLabelImg * 0.8 + detectRes * 0.2;

			// cv::Mat h1, h2,h3, res;
			// vconcat(visRgb, visDepth, h1);
			// vconcat(detectRes, phiMap, h2);
			// vconcat(classLabelImg, kaiMap, h3);
			// hconcat(h1, h2, res);
			// hconcat(res, h3, res);
			// cv::imshow("res", res);
			// string output_path = "./tmp/" + dataset_name + "/" + dataEngine->getRGBFile(sidx, fidx);
			//cv::imwrite(output_path, res);


			imshow("mask", mask);
			
			imshow("label", labelImg);

			//photoSegm_thread.join();
			// cv::Mat photoSegmImg = photoSegmEngine->getsegmImg();
			// cv::imshow("photosegm", photoSegmImg);
			
			int key = waitKey(1);
			if(key == 'q')
				break;
			clock_t end = clock();
			cout << "FPS = " << CLOCKS_PER_SEC / (double)(end - start) << endl;
			//delete geoSegMap_Host;
			//delete objSegMap_Host;
			// Mat h1, h2, res;
			// hconcat(rgb, detectRes, h1);
			// hconcat(newNorm, mask, h2);
			// vconcat(h1, h2, res);
			// writer << res;
		}
		//writer.release();
		break;
	}

}