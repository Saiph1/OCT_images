pre:
	pip install -U openmim
	python -m pip install -r requirements.txt
	mim install mmcv-full==1.7.1
install:
	make pre
	git clone https://github.com/microsoft/SoftTeacher.git
	pip install -v -e ./SoftTeacher
	git clone https://github.com/open-mmlab/mmdetection.git ./SoftTeacher/thirdparty/mmdetection
	cd SoftTeacher/thirdparty/mmdetection && git checkout v2.28.1 && python -m pip install -e .
	pip install -v -e ./SoftTeacher/thirdparty/mmdetection