if [ -d "build" ]
then 
    echo "build directory exists. removing build folder."
	echo "starting build process..."
	rm -rf build
	mkdir build && cd build
	cmake -G "Visual Studio 15 2017" ..
	cmake --build .
	echo "executable can be found in build/debug"

else
    echo "starting build process..."
    mkdir build && cd build
	cmake -G "Visual Studio 15 2017" ..
	cmake --build .
	echo "executable can be found in build/debug"
fi
