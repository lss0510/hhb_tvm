#include<opencv2/opencv.hpp>
#include<iostream>
#include<fstream>

using namespace cv;

void load_image_and_preprocess()
{
  // load image
  Mat origin_img = imread("persian_cat.jpg");

  int image_width = 224;
  int image_height = 224;
  int image_channel = 3;

  // resize image to 224 * ? or ? * 224
  cv::Mat resized_img;
  float ratio = 224 / fmin(origin_img.size().width, origin_img.size().height);
  cv::resize(origin_img, resized_img, cv::Size(), ratio, ratio);

  // Crop image center 224 * 224
  int start_x = resized_img.size().width / 2 - image_width / 2;
  int start_y = resized_img.size().height / 2 - image_height / 2;
  cv::Rect crop_region(start_x, start_y, image_width, image_height);
  cv::Mat cropped_image = resized_img(crop_region);

  // convert to float
  cv::Mat float_img;
  cropped_image.convertTo(float_img, CV_32F);

  // bgr to rgb
  cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB);

  // mean
  float_img -= Scalar(124, 117, 104);

  // scale
  float_img *= 0.017;

  // save to file
  FILE* fp = fopen("input_img.tensor", "w");
  FILE* bfp = fopen("input_img.bin", "w");

  float *f32_ptr = float_img.ptr<float>(0);
  float float_data[image_channel * image_width * image_height];

  // layout to be CHW
  for (int k = 0; k < image_channel ; k++) {
    for (int i = 0; i < image_width * image_height; i++) {
      float point = f32_ptr[k + i * image_channel];
      fprintf(fp, "%f\n", point);
      float_data[k * image_width * image_height + i] = point;
    }
  }

  fwrite(float_data, sizeof(float), image_channel * image_width * image_height, bfp);

  fclose(fp);
  fclose(bfp);
}

static void get_top5(float *buf, uint32_t size, float *prob, uint32_t *cls)
{
  uint32_t i, j, k;

  memset(prob, 0xfe, sizeof(float) * 5);
  memset(cls, 0xff, sizeof(uint32_t) * 5);

  for (j = 0; j < 5; j++)
  {
    for (i = 0; i < size; i++)
    {
      for (k = 0; k < 5; k++)
      {
        if (i == cls[k])
        {
          break;
        }
      }

      if (k != 5)
      {
        continue;
      }

      if (buf[i] > prob[j])
      {
        prob[j] = buf[i];
        cls[j] = i;
      }
    }
  }
}

static float* get_data_from_file(const char* filename, uint32_t size) {
  uint32_t j;
  float fval = 0.0;
  float* buffer = NULL;
  FILE* fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("Invalid input file: %s\n", filename);
    return NULL;
  }

  buffer = (float *)malloc(size * sizeof(float));
  if (buffer == NULL) {
    printf("Malloc fail\n");
    return NULL;
  }
  for (j = 0; j < size; j++) {
    if (fscanf(fp, "%f ", &fval) != 1) {
      printf("Invalid input size\n");
      return NULL;
    } else {
      buffer[j] = fval;
    }
  }

  fclose(fp);
  return buffer;
}

void load_result_and_postprocess()
{
  uint32_t i = 0, size = 32;
  uint32_t cls[5];
  float prob[5];

  float* output_data = get_data_from_file("input_img.bin_output0_1_32.txt", 32);

  get_top5(output_data, size, prob, cls);

  std::ifstream infile;
  infile.open("synset.txt");
  std::vector<std::string> labels;
  std::string line;
  while (getline(infile, line))
  {
    labels.push_back(line);
  }

  std::cout << " ********** probability top5: ********** " << std::endl;
  size = size > 5 ? 5 : size;
  for (i = 0; i < size; i++)
  {
    std::cout << labels[cls[i]] << std::endl;
  }
}

int main()
{
  std::cout << " ********** preprocess image **********" << std::endl;
  load_image_and_preprocess();

  std::cout << " ********** run lss test_model **********" << std::endl;
  system("./hhb_out_h/hhb_runtime ./hhb_out_h/hhb.bm input_img.bin");

  std::cout << " ********** postprocess result **********" << std::endl;
  load_result_and_postprocess();

  return 0;
}