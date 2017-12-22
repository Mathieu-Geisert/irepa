IREPA: Python script and ACADO C++ binaries to code the Iterative Roadmap-Extension and Policy-Approximation (IRPA) algorithm
=====

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE 
make connect_quadcopter_pendulum
cd ../python
ipython -i plearn_quadcopterpendulum.py 


