## Training with Augumentation

Good

```
Detecting for class test-tench.jpg model resnet50
--------------------------------
tench 0.999745786190033
English springer 0.00025274601648561656
golf ball 6.280536695157934e-07
garbage truck 4.128370107991941e-07
cassette player 2.1240308001324593e-07
--------------------------------
Detecting for class test-church.jpg model resnet50
--------------------------------
church 0.9999773502349854
gas pump 1.3593445146398153e-05
garbage truck 8.783520570432302e-06
cassette player 1.7043690547779988e-07
parachute 9.653887644844872e-08
--------------------------------
Detecting for class test-garbagetruck.jpg model resnet50
--------------------------------
garbage truck 0.9934307932853699
gas pump 0.0064073423855006695
chain saw 7.451870624208823e-05
church 7.211051706690341e-05
golf ball 8.967723260866478e-06
--------------------------------
Detecting for class test-truck.jpg model resnet50
--------------------------------
garbage truck 0.999943733215332
gas pump 4.955384792992845e-05
parachute 4.725446615339024e-06
chain saw 9.217237106895482e-07
cassette player 5.909126912229112e-07
--------------------------------
Detecting for class test-dog.jpg model resnet50
--------------------------------
English springer 0.9841727018356323
golf ball 0.015795063227415085
parachute 1.535027513455134e-05
tench 1.4700498468300793e-05
chain saw 1.1618815278779948e-06
--------------------------------
Detecting for class test-englishspringer.jpeg model resnet50
--------------------------------
English springer 0.999875545501709
church 7.528714195359498e-05
chain saw 4.66570163553115e-05
golf ball 1.6838457668200135e-06
parachute 2.894360022764886e-07
--------------------------------
````

Model RestNet50_11:43_October072022.pth
Trainined with 1000 odd dog images

2022-10-07 11:43:06,459 --->Epoch [20/20], Average Loss: 0.3130 Average Accuracy: 89.5503
Accuracy of the network on the 3925 test/validation images: 73.93630573248407 %
Accuracy of the network on the 12198 Train images: 89.5556648630923 %

```
Detecting for class test-tench.jpg model resnet50
--------------------------------
tench 0.9999969005584717
garbage truck 1.8156019905291032e-06
chain saw 7.439767841788125e-07
English springer 4.304067147131718e-07
cassette player 6.587828238480142e-08
--------------------------------
Detecting for class test-church.jpg model resnet50
--------------------------------
church 0.9980545043945312
garbage truck 0.0017299364553764462
gas pump 0.00015054643154144287
cassette player 2.814459912769962e-05
parachute 2.723202487686649e-05
--------------------------------
Detecting for class test-garbagetruck.jpg model resnet50
--------------------------------
garbage truck 0.9999949932098389
gas pump 3.8616653910139576e-06
chain saw 4.952198082719406e-07
cassette player 4.744395027955761e-07
English springer 1.418114266016346e-07
--------------------------------
Detecting for class test-truck.jpg model resnet50
--------------------------------
garbage truck 0.9999574422836304
chain saw 2.3277179934666492e-05
gas pump 8.556658031011466e-06
English springer 5.3411140470416285e-06
cassette player 4.366767370811431e-06
--------------------------------
Detecting for class test-dog.jpg model resnet50
--------------------------------
golf ball 0.6378457546234131        -------------> Bad            
English springer 0.3451835513114929
tench 0.010568302124738693
parachute 0.0033155360724776983
chain saw 0.0013443181524053216
--------------------------------
Detecting for class test-englishspringer.jpg model resnet50
--------------------------------
chain saw 0.684698224067688           ---------------> Bad
French horn 0.19285185635089874
English springer 0.035448700189590454
cassette player 0.031124675646424294
tench 0.024074256420135498
--------------------------------
Detecting for class test_dogcartoon.jpg model resnet50
--------------------------------
golf ball 0.3991181552410126         -----------------> Bad
cassette player 0.3253536522388458
chain saw 0.18893717229366302
English springer 0.07033721357584
garbage truck 0.008056157268583775
--------------------------------
```