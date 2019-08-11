# TensorFlow C API

TensorFlow lib C API on Linux

## [Example](src/)

* [Hello TF](src/hello_tf.cpp)
* [Load graph](src/load_graph.cpp)
* [Create Tensor](src/create_tensor.cpp)
* [Allocate Tensor](src/allocate_tensor.cpp)
* [Run session](src/session_run.cpp)
* [Interface](src/interface.cpp)
* [Tensor Info](src/tensor_info.cpp)
* [Graph Info](src/graph_info.cpp)

## Build example

### Linux

```text
git clone --depth 1 https://github.com/Neargye/hello_tf_c_api
cd hello_tf_c_api
mkdir build
cd build
cmake -G "Unix Makefiles" ..
cmake --build .
```

### Link tensorflow lib

#### CMakeLists.txt

```text
link_directories(yourpath/to/tensorflow) # path to tensorflow lib
... # other
target_link_libraries(<target> <PRIVATE|PUBLIC|INTERFACE> tensorflow)
```

### [Here’s an example how prepare models](doc/prepare_models.md)

To generated the graph.pb file need takes a graph definition and a set of checkpoints and freezes them together into a single file.


## Licensed under the [MIT License](LICENSE)
