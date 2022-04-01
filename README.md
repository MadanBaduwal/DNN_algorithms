# DNN algorithms

# **If you want to find the secret of computer science think in term of matrix(meaning of row and column)**

* **Express everything in datapoints(english,physics,chemistry,bilogy,engineering)> then express in computer science concept matrix(number>datapoint(vector)>matrix>tensor) > then express mathematics concept in matrix linear algebra(points plot in space),probability and statistics(probability and statistics of these data points),calcalus(change rate of these data points)...**

* row = number of datapoints

* column = dimension of datapoint


| input things | feature1 | feature2|feature3|...|Label(y)|
| --- | --- | --- |---|---|---|
| datapoint1(x_1)(image1/Sound1/word1/sentence1/english/math/physics/chemistry/biology...Everything into data point) | value | value | value|...| cat/"hello"...|
| datapoint2(x_2)(image2/Sound2/word2/sentence2/english/math/physics/chemistry/biology...Everything into data point) | value | value | value|...| dog/"how"...|
| datapoint3(x_3)(image3/Sound3/word3/sentence3/english/math/physics/chemistry/biology...Everything into data point) | value | value | value|...| cow/"are"...|
| datapoint4(x_4)(image4/Sound4/word4/sentence4/english/math/physics/chemistry/biology...Everything into data point) | value | value | value|...| elephant/"you"...|

    [image1/Sound1/word1/sentence1/english/math/physics/chemistry/biology...Everything into data point ] 
    [ image2/Sound2/word2/sentence2/english/math/physics/chemistry/biology...Everything into data point ] 
    [image3/Sound3/word3/sentence3/english/math/physics/chemistry/biology...Everything into data point ]
    [ image4/Sound4/word4/sentence4/english/math/physics/chemistry/biology...Everything into data ]


        [1  2  1 ] 
    x = [3  0  1 ]
        [0  2  4 ]
        [0  2  4 ]

        [ cat/"hello"...]
    y = [ dog/"how"...  ]
        [ cow/"are"... ]
        [ elephant/"you"...]



* **Always imagine data point(multi dimensional data point) in space and give this data point to label**.

* **Anything is number and this number carry some information**


## **AND**

### **1.Any Programming language project structure**

```
├── Package               <- Folder Contain Many programming files(modules)
    ├── Module            <- File (i.e  .h(in c and c++),.py,.js ,.java,php...)
       ├── import from another package and module < -#include ... in c and c++, import in python and java
       ├── Module attribute
       ├── Class         
       │   ├── Class attribute     
       │   ├── Method      
       │         ├── Method attribute 
       ├── Function
           ├── Function attribute
       ├── Generator
           ├── Generator attribute

```



# Table of content

-------

1. Leneat
2. Alexnet
3. Attention is all you need
4. Swin transformer






