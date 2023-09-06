pre:
	pip install -U openmim
	mim install mmengine
	python -m pip install -r requirements.txt
	git clone https://github.com/open-mmlab/mmdetection.git
	cd mmdetection && git checkout v2.28.1 && python -m pip install -e .
	pip install mmcv-full=1.7.1
install:
	make pre
	cd SoftTeacher
	python -m pip install -e .

