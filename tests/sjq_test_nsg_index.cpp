//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <rust-lib.h>
#include "omp.h"
int main(int argc, char **argv)
{

#pragma omp parallel
    {
        std::vector<int> v;

#pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < 100; i++)
        {
            sleep(2);
            auto thread_idx = omp_get_thread_num();
            std::cout << "i: "<<i<<" thread: "<<thread_idx<<std::endl;
        }
    }

    return 0;
}