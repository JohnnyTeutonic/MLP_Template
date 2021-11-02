if [ -d "build_linux" ]
then 
    echo "build directory exists. removing build folder."
	rm -rf build_linux
fi
    echo "making build directory"
    mkdir build_linux && cd build_linux
    echo "starting build process..."
    cmake .. -G"Unix Makefiles"
    make
    echo "executable can be found in \"build_linux\" named updated_mlp"
