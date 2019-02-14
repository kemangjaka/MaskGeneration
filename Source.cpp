#include "Header.h"
#include "./mainEngine/mainEngine.h"

int main(int argc, char *argv[])
{
	if (argc == 1)
	{
		cout << "Usage : ./MaskGeneration {dataset_id}" << endl;
		cout << "RGBD scene dataset : 0" << endl;
		cout << "NYU v2 dataset : 1" << endl;
		cout << "co-fusion clock :2" << endl;
		return 1;
	}
	else
	{
		char *p;
		int dataset_id = strtol(argv[1], &p, 10);
		if (dataset_id >= 0 && dataset_id < 3)
		{
			MainEngine engine(dataset_id);
			engine.Activate();
		}
	}
}