
#include <iostream>
#include <fstream>
int main()
{
    u_int32_t dim;
    u_int32_t num;
    float *data;
    auto filename = "/home/sjq/git/nsg-fork/gist_base.fvecs";
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *)&dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new float[(size_t)num * (size_t)dim];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++)
    {
        in.seekg(4, std::ios::cur);
        in.read((char *)(data + i * dim), dim * 4);
    }
    in.close();
    std::cout << "dim: " << dim << " num: " << num << std::endl;
    for (int i = 0; i < 10; i++)
    {
        std::cout << data[i] << std::endl;
    }
    for (int i = (1000000 - 1) * 960; i < (1000000 - 1) * 960 + 10; i++)
    {
        std::cout << data[i] << std::endl;
    }

    delete[] data;
}