# Overview
This repository contains code for the paper:

Bryan Wilder, Yevgeniy Vorobeychik. Defending Elections Against Malicious Spread of Misinformation. AAAI Conference on Artificial Intelligence. 2019.
```
@inproceedings{wilder2019defending,
 author = {Wilder, Bryan and Vorobeychik, Yevgeniy},
 title = {Defending Elections Against Malicious Spread of Misinformation},
 booktitle = {Proceedings of the 33rd AAAI Conference on Artificial Intelligence},
 year = {2019}
}
```

Included is the code to run online gradient descent for the nondisjoint case considered in the paper. The dataset was based on the [Yahoo webscope search marketing advertiser bidding dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=a&guccounter=1&guce_referrer=aHR0cHM6Ly93ZWJzY29wZS5zYW5kYm94LnlhaG9vLmNvbS8&guce_referrer_sig=AQAAACUqCqOIRKcfRKf4YWuAGAikBI35GlG5K57eaK2cxPpEfeKv_RdXxQl70q1DtbxMovefEnEOACXSrgU9IB8oHPqEsRjzHEFBN4LtLOjV9DSbeeT3JVgJX3D_7nV52YZKATMXkLdTS1F5h60idD5sUnHxY_uRaKuNfSjNDXuYMur_), which can be accessed by submitting a request to Yahoo. Regardless of source, the code expects as input a numpy representation of the adjacency matrix of the bipartite graph. This is a matrix P where P[i,j] gives the weight on the (i,j) edge.
