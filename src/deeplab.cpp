#include <c_api.h> // TensorFlow C API header
#include <c_api_experimental.h>
#include <scope_guard.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>

// https://imagemagick.org/api/magick-image.php
#include <wand/magick_wand.h>

#include "tf_utils.hpp"

static void DeallocateBuffer(void* data, size_t) {
  std::free(data);
}

static TF_Buffer* ReadBufferFromFile(const char* file) {
  std::ifstream f(file, std::ios::binary);
  SCOPE_EXIT{ f.close(); };
  if (f.fail() || !f.is_open()) {
    return nullptr;
  }

  f.seekg(0, std::ios::end);
  auto fsize = f.tellg();
  f.seekg(0, std::ios::beg);

  if (fsize < 1) {
    return nullptr;
  }

  auto data = static_cast<char*>(std::malloc(fsize));
  f.read(data, fsize);

  auto buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = DeallocateBuffer;

  return buf;
}

int main() {

	MagickWand *mw = NULL;

	MagickWandGenesis();


	mw = NewMagickWand();

	/* Read the input image */
	MagickReadImage(mw,"sample.jpg");
  MagickScaleImage(mw, 512, 288);

  std::vector<std::uint8_t> pixels;
  pixels.resize(512 * 288 * 3);
  MagickExportImagePixels(mw, 0, 0, 512, 288, "RGB", CharPixel, pixels.data());
  // std::cout << "pixels: " << pixels << std::endl;


  // mw = NewMagickWand();
  // MagickImportImagePixels(mw, 0, 0, 512, 288, "RGB", CharPixel, pixels);
  // size_t image_buffer_size;
  // // MagickSetFormat(mw, "RAW");
  // unsigned char* image_buffer = MagickGetImageBlob(mw, &image_buffer_size);
  // std::cout << "Image" << " size " << image_buffer_size << std::endl;
	// MagickWriteImage(mw,"sample.png");


	MagickWandTerminus();

  std::cout << TF_Version() << std::endl;


  auto graph_buffer = ReadBufferFromFile("frozen_inference_graph.pb");
  if (graph_buffer == nullptr) {
    std::cout << "Can't read buffer from file" << std::endl;
    return 1;
  }

  TF_Graph* graph = TF_NewGraph();
  TF_Status* status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };
  auto opts = TF_NewImportGraphDefOptions();

  TF_GraphImportGraphDef(graph, graph_buffer, opts, status);
  TF_DeleteImportGraphDefOptions(opts);
  TF_DeleteBuffer(graph_buffer);

  if (TF_GetCode(status) != TF_OK) {
    TF_DeleteGraph(graph);
    std::cout << "Can't import GraphDef" << std::endl;
    return 2;
  }

  std::cout << "Load graph success" << std::endl;

  auto input_op = TF_Output{TF_GraphOperationByName(graph, "ImageTensor"), 0};
  if (input_op.oper == nullptr) {
    std::cout << "Can't init input_op" << std::endl;
    return 2;
  }

  auto output_op = TF_Output{TF_GraphOperationByName(graph, "SemanticPredictions"), 0};
  if (output_op.oper == nullptr) {
    std::cout << "Can't init output_op" << std::endl;
    return 2;
  }

  const std::vector<std::int64_t> input_dims = {1, 512, 288, 3};
  TF_Tensor* input_tensor = tf_utils::CreateTensor(TF_UINT8, input_dims, pixels);
  SCOPE_EXIT{ tf_utils::DeleteTensor(input_tensor); };

  const std::vector<std::int64_t> output_dims = {1, 512, 288};
  TF_Tensor* output_tensor = {tf_utils::CreateEmptyTensor(TF_INT64, output_dims)};
  SCOPE_EXIT{ tf_utils::DeleteTensor(output_tensor); };

  // auto options = tf_utils::CreateSessionOptions(100, status);
  // if (TF_GetCode(status) != TF_OK) {
  //   std::cout << "Error CreateSessionOptions";
  //   return 5;
  // }

  auto options = TF_NewSessionOptions();
  // TF_EnableXLACompilation(options, true);

  auto sess = TF_NewSession(graph, options, status);
  TF_DeleteSessionOptions(options);

  if (TF_GetCode(status) != TF_OK) {
    return 4;
  }

  TF_SessionRun(sess,
                nullptr, // Run options.
                &input_op, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
                &output_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
                nullptr, 0, // Target operations, number of targets.
                nullptr, // Run metadata.
                status // Output status.
                );

  // std::cout << "STATUS: " << TF_Message(status->get()) << std::endl;

  if (TF_GetCode(status) != TF_OK) {
    std::cout << "Error run session";
    return 5;
  }

  TF_CloseSession(sess, status);
  if (TF_GetCode(status) != TF_OK) {
    std::cout << "Error close session";
    return 6;
  }

  TF_DeleteSession(sess, status);
  if (TF_GetCode(status) != TF_OK) {
    std::cout << "Error delete session";
    return 7;
  }

  auto data = static_cast<std::int64_t*>(TF_TensorData(output_tensor));

  // data[512] = 3;
  // data[612] = 2;
  // data[712] = 1;

  std::vector<std::int64_t> output;
  output.assign(data, data + 512 * 288);

  std::cout << " max element is: " << *std::max_element(data , data + 512 * 288) << std::endl;
  std::cout << "Output max_value: " << *std::max_element(output.begin(), output.end()) << std::endl;

  return 0;
}

