
##################################################################
##################Table 2.3, 2.4 on 9x9, 19x19#################
#################################################################
START_LOGS= 1
run:
	python run_multiple_experiments.py --num_epochs 400 --Model_type conditional --Model_structure FF --categorization 2 --num_repeats 5 --Pixel_Full 9
	python Results.py --model_name 'cGAN_9x9' --filename START_LOGS --iterations 5
START_LOGS= 6
run:
	python run_multiple_experiments.py --num_epochs 400 --Model_type conditional --Model_structure Convolutional --categorization 2 --num_repeats 5 --Pixel_Full 9
	python Results.py --model_name 'cCGAN_9x9' --filename START_LOGS --iterations 5
START_LOGS= 11
run:
	python run_multiple_experiments.py --num_epochs 400 --Model_type conditional --Model_structure FF --categorization 2.3 --num_repeats 5 --Pixel_Full 19
	python Results.py --model_name 'cGAN_19x19' --filename START_LOGS --iterations 5
START_LOGS= 16
run:
	python run_multiple_experiments.py --num_epochs 400 --Model_type conditional --Model_structure Convolutional --categorization 2.3 --num_repeats 5 --Pixel_Full 19
	python Results.py --model_name 'cCGAN_19x19' --filename START_LOGS --iterations 5
##################################################################
##################Table  2.5, 2.6, 2.7 on 6 and 8 categories ##################
#################################################################

START_LOGS= 21
run:
	python run_multiple_experiments.py --num_epochs 400 --Model_type conditional --Model_structure FF --categorization 6 --num_repeats 5 --Pixel_Full 9
	python Results.py --model_name 'cGAN_9x9_6Cat' --filename START_LOGS --iterations 5

START_LOGS= 26
run:
	python run_multiple_experiments.py --num_epochs 400 --Model_type conditional --Model_structure Convolutional --categorization 6 --num_repeats 5 --Pixel_Full 9
	python Results.py --model_name 'cCGAN_9x9_6Cat' --filename START_LOGS --iterations 5

START_LOGS= 21
run:
	python run_multiple_experiments.py --num_epochs 400 --Model_type conditional --Model_structure FF --categorization 8 --num_repeats 5 --Pixel_Full 9
	python Results.py --model_name 'cGAN_9x9_8Cat' --filename START_LOGS --iterations 5

START_LOGS= 26
run:
	python run_multiple_experiments.py --num_epochs 400 --Model_type conditional --Model_structure Convolutional --categorization 8 --num_repeats 5 --Pixel_Full 9
	python Results.py --model_name 'cCGAN_9x9_8Cat' --filename START_LOGS --iterations 5
clean:
	rm -rf __pycache__
.PHONY:	run clean