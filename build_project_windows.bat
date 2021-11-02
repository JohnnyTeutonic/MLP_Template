if not exist "build" ( 
    echo Creating build folder...
    mkdir build
)

if exist "build" ( 
    echo Re-Creating build Folder...
    rmdir /s /q build
)
echo "starting build process..."
cd build
cmake -G "Visual Studio 15 2017" ..
cmake --build .
echo "executable can be found in "build/debug" named updated_mlp.exe"
	