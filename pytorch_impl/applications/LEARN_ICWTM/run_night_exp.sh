./run_exp.sh resnet18 | tee -a logFile_night_exp_neurips20_sub
sleep 10800
./kill.sh
sleep 60
./run_exp.sh resnet34 | tee -a logFile_night_exp_neurips20_sub
sleep 10800
./kill.sh
sleep 60
./run_exp.sh resnet50 | tee -a logFile_night_exp_neurips20_sub
sleep 10800
./kill.sh
sleep 60


