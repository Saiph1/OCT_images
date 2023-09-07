pre:
	pip install -U openmim
	mim install mmengine
	python -m pip install -r requirements.txt
	pip install mmcv-full==1.7.1
	git clone https://github.com/open-mmlab/mmdetection.git
	cd mmdetection && git checkout v2.28.1 && python -m pip install -e .
install:
	make pre
	git clone https://github.com/microsoft/SoftTeacher.git
	pip install -v -e ./SoftTeacher