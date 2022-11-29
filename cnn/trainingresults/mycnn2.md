
model: mycnn2_19:22_October142022.pth

```
2022-10-15 19:12:38,150 --->Epoch [20/20], Average Loss: 0.6561 Average Accuracy: 79.1632
Accuracy of the network on the 3925 test/validation images: 63.69426751592356 %
Accuracy of the network on the 9476 Train images: 82.01772899957788 %
```
Test output

```
Detecting for class test-tench.jpg model mycnn2
--------------------------------
tench 0.9070395827293396
chain saw 0.07627348601818085
English springer 0.008709638379514217
gas pump 0.0058251009322702885
church 0.0011570182396098971
--------------------------------
Detecting for class test-church.jpg model mycnn2
--------------------------------
church 0.9998339414596558
French horn 0.0001568781299283728
gas pump 6.263525847316487e-06
garbage truck 1.4481427115242695e-06
parachute 6.459334827013663e-07
--------------------------------
Detecting for class test-garbagetruck.jpg model mycnn2
--------------------------------
garbage truck 0.6557585000991821
French horn 0.17168129980564117
church 0.1694873869419098
gas pump 0.0025451709516346455
chain saw 0.00044949323637411
--------------------------------
Detecting for class test-truck.jpg model mycnn2
--------------------------------
garbage truck 0.9464776515960693
parachute 0.0196551401168108
gas pump 0.015828553587198257
French horn 0.005917742382735014
chain saw 0.005670647602528334
--------------------------------
Detecting for class test-dog.jpg model mycnn2
--------------------------------
golf ball 0.5694670081138611 ----------------------------------->bad
English springer 0.4237217903137207
tench 0.006679318379610777
parachute 4.9689453589962795e-05
gas pump 3.8956237403908744e-05
--------------------------------
Detecting for class test-englishspringer.jpg model mycnn2
--------------------------------
English springer 0.4231145977973938
French horn 0.30285757780075073
golf ball 0.14319057762622833
chain saw 0.09181413799524307
church 0.022737208753824234
--------------------------------
Detecting for class test_dogcartoon.jpg model mycnn2
--------------------------------
chain saw 0.5923001170158386 ----------------------------------->bad
golf ball 0.2379758059978485
French horn 0.09823056310415268
church 0.02583104558289051
English springer 0.01899402029812336
--------------------------------
Detecting for class test_chaingsaw.jpg model mycnn2
--------------------------------
French horn 0.5273041129112244
chain saw 0.27078473567962646
garbage truck 0.09268984198570251
English springer 0.03983509913086891
gas pump 0.02689962461590767
--------------------------------
Detecting for class test_chainsawtrain.jpg model mycnn2
--------------------------------
chain saw 0.4872604012489319
French horn 0.42065727710723877
gas pump 0.07386265695095062
English springer 0.013915135525166988
church 0.001717262901365757
--------------------------------
Detecting for class test_frenchhorn.jpg model mycnn2
--------------------------------
French horn 0.9959678649902344
tench 0.0024671771097928286
golf ball 0.0007324784528464079
chain saw 0.00026514698402024806
cassette player 0.00021084898617118597
--------------------------------
Detecting for class test_frenchhorntrain.jpg model mycnn2
--------------------------------
French horn 0.9451285004615784
gas pump 0.021555252373218536
church 0.013406408950686455
chain saw 0.0066641345620155334
golf ball 0.005017921328544617
--------------------------------
Detecting for class test-golfball.jpg model mycnn2
--------------------------------
golf ball 0.9922829270362854
English springer 0.005850684829056263
parachute 0.0006327238515950739
gas pump 0.00045258254976943135
chain saw 0.000365490501280874
--------------------------------
```

Adding one more layer

```
2022-10-18 11:56:25,880 --->Epoch [20/20], Average Loss: 0.6267 Average Accuracy: 79.4778
Accuracy of the network on the 3925 test/validation images: 68.6624203821656 %
Accuracy of the network on the 9476 Train images: 82.83030814689742 %
```

```
Detecting for class test-tench.jpg model mycnn2
--------------------------------
tench 0.6015536785125732
English springer 0.39521777629852295
golf ball 0.001436604536138475
chain saw 0.0009375310619361699
parachute 0.0006053766701370478
--------------------------------
Detecting for class test-church.jpg model mycnn2
--------------------------------
church 0.997459352016449
gas pump 0.001410320051945746
garbage truck 0.000522431218996644
parachute 0.0003147284733131528
chain saw 0.00014782557263970375
--------------------------------
Detecting for class test-garbagetruck.jpg model mycnn2
--------------------------------
garbage truck 0.9380949139595032
gas pump 0.06073734909296036
church 0.000751270679756999
cassette player 0.00022187209106050432
parachute 0.00013170672173146158
--------------------------------
Detecting for class test-truck.jpg model mycnn2
--------------------------------
garbage truck 0.7907505035400391
cassette player 0.18294143676757812
chain saw 0.009447077289223671
French horn 0.006453351117670536
gas pump 0.005344416480511427
--------------------------------
Detecting for class test-dog.jpg model mycnn2
--------------------------------
English springer 0.8120921850204468
golf ball 0.18565377593040466
parachute 0.0019213404739275575
church 0.00013716051762457937
chain saw 0.0001059037385857664
--------------------------------
Detecting for class test-englishspringer.jpg model mycnn2
--------------------------------
English springer 0.35248300433158875
French horn 0.2586326003074646
chain saw 0.19894850254058838
golf ball 0.10888627171516418
church 0.05291454866528511
--------------------------------
Detecting for class test_dogcartoon.jpg model mycnn2
--------------------------------
chain saw 0.42451855540275574 ---------------------------------->bad
golf ball 0.19405700266361237
parachute 0.17442800104618073
cassette player 0.09111572802066803
French horn 0.07398094236850739
--------------------------------
Detecting for class test_chaingsaw.jpg model mycnn2
--------------------------------
chain saw 0.6011797785758972
garbage truck 0.1872745156288147
gas pump 0.11830461025238037
English springer 0.023241812363266945
church 0.022928878664970398
--------------------------------
Detecting for class test_chainsawtrain.jpg model mycnn2
--------------------------------
chain saw 0.8492781519889832
English springer 0.07455718517303467
gas pump 0.03897153213620186
French horn 0.02428496815264225
church 0.008693202398717403
--------------------------------
Detecting for class test_frenchhorn.jpg model mycnn2
--------------------------------
French horn 0.9922921657562256
golf ball 0.0033552309032529593
tench 0.0013955659233033657
chain saw 0.001253218506462872
church 0.0009560852777212858
--------------------------------
Detecting for class test_frenchhorntrain.jpg model mycnn2
--------------------------------
French horn 0.6674312353134155
church 0.1692582666873932
gas pump 0.13076777756214142
chain saw 0.01796167530119419
English springer 0.00577181251719594
--------------------------------
Detecting for class test-golfball.jpg model mycnn2
--------------------------------
golf ball 0.786182165145874
chain saw 0.1236642375588417
parachute 0.026523718610405922
English springer 0.0240766741335392
church 0.01588711142539978
--------------------------------
```

Changed training for dog classes to includes varied dogs

```
Accuracy of the network on the 3925 test/validation images: 63.745222929936304 %
Accuracy of the network on the 9689 Train images: 77.61378883269687 %
```


```
Detecting for class test-tench.jpg model mycnn2
--------------------------------
tench 0.9892221689224243
chain saw 0.008946406655013561
English springer 0.0010823128977790475
parachute 0.0005199289880692959
gas pump 0.00012892343511339277
--------------------------------
Detecting for class test-church.jpg model mycnn2
--------------------------------
church 0.9811632037162781
French horn 0.01492960937321186
gas pump 0.0026152811478823423
garbage truck 0.0006906508933752775
chain saw 0.000519688066560775
--------------------------------
Detecting for class test-garbagetruck.jpg model mycnn2
--------------------------------
garbage truck 0.9686723947525024
gas pump 0.028597809374332428
cassette player 0.0013998955255374312
church 0.0006609823321923614
French horn 0.0005822560633532703
--------------------------------
Detecting for class test-truck.jpg model mycnn2
--------------------------------
garbage truck 0.9824532270431519
French horn 0.007712746039032936
cassette player 0.0072258394211530685
chain saw 0.0018852191278710961
parachute 0.00027308118296787143
--------------------------------
Detecting for class test-dog.jpg model mycnn2
--------------------------------
golf ball 0.9038493633270264
English springer 0.05762539431452751
tench 0.015439094044268131
parachute 0.009726260788738728
chain saw 0.008810309693217278
--------------------------------
Detecting for class test-englishspringer.jpg model mycnn2
--------------------------------
English springer 0.824559211730957
chain saw 0.1709856539964676
church 0.002131542656570673
French horn 0.0020524782594293356
golf ball 0.00012985875946469605
--------------------------------
Detecting for class test_dogcartoon.jpg model mycnn2
--------------------------------
French horn 0.4396744668483734
golf ball 0.365559458732605
parachute 0.08742266893386841
chain saw 0.08176645636558533
cassette player 0.021988680586218834
--------------------------------
Detecting for class test_chaingsaw.jpg model mycnn2
--------------------------------
chain saw 0.9090695381164551
French horn 0.0333731546998024
tench 0.027559000998735428
garbage truck 0.014986671507358551
English springer 0.012394114397466183
--------------------------------
Detecting for class test_chainsawtrain.jpg model mycnn2
--------------------------------
chain saw 0.794063150882721
French horn 0.11026885360479355
church 0.06919961422681808
English springer 0.01739812083542347
gas pump 0.006316084414720535
--------------------------------
Detecting for class test_frenchhorn.jpg model mycnn2
--------------------------------
French horn 0.989661455154419
chain saw 0.007443428970873356
tench 0.002476726658642292
English springer 0.00020236472482793033
church 0.00011428951256675646
--------------------------------
Detecting for class test_frenchhorntrain.jpg model mycnn2
--------------------------------
French horn 0.9052383899688721
church 0.042890969663858414
English springer 0.029640385881066322
chain saw 0.015361960977315903
gas pump 0.004605152644217014
--------------------------------
Detecting for class test-golfball.jpg model mycnn2
--------------------------------
golf ball 0.8072097301483154
English springer 0.175912007689476
chain saw 0.007657335605472326
church 0.006996778771281242
parachute 0.0011426492128521204
```

2022-10-18 16:20:54,936 --->Epoch [20/20], Average Loss: 0.2965 Average Accuracy: 90.4962
Accuracy of the network on the 3925 test/validation images: 63.64331210191083 %
Accuracy of the network on the 9694 Train images: 92.42830616876418 %


Accuracy of the network on the 3925 test/validation images: 64.07643312101911 %
Accuracy of the network on the 9694 Train images: 72.59129358365999 %

```
Detecting for class test-tench.jpg model mycnn2
--------------------------------
tench 0.891904890537262
chain saw 0.06602543592453003
English springer 0.03672690689563751
golf ball 0.002956508193165064
garbage truck 0.0011928387684747577
--------------------------------
Detecting for class test-church.jpg model mycnn2
--------------------------------
church 0.9892656803131104
gas pump 0.005277703981846571
garbage truck 0.0049094040878117085
golf ball 0.00027121484163217247
parachute 0.00012306208373047411
--------------------------------
Detecting for class test-garbagetruck.jpg model mycnn2
--------------------------------
garbage truck 0.7443642020225525
gas pump 0.21422001719474792
church 0.01523587666451931
cassette player 0.012792576104402542
chain saw 0.008517378941178322
--------------------------------
Detecting for class test-truck.jpg model mycnn2
--------------------------------
garbage truck 0.8158890604972839
cassette player 0.07770562171936035
chain saw 0.06115741282701492
gas pump 0.01694645918905735
parachute 0.010777538642287254
--------------------------------
Detecting for class test-dog.jpg model mycnn2
--------------------------------
English springer 0.6350854635238647
golf ball 0.2736964225769043
tench 0.05095984786748886
chain saw 0.030257880687713623
parachute 0.008038467727601528
--------------------------------
Detecting for class test-englishspringer.jpg model mycnn2
--------------------------------
golf ball 0.8366643786430359
English springer 0.10515034943819046
French horn 0.02347155660390854
chain saw 0.023351605981588364
church 0.0069016325287520885
--------------------------------
Detecting for class test_dogcartoon.jpg model mycnn2
--------------------------------
golf ball 0.6620281934738159
chain saw 0.21367815136909485
cassette player 0.04637474566698074
parachute 0.03648553043603897
English springer 0.023071512579917908
--------------------------------
Detecting for class test_chaingsaw.jpg model mycnn2
--------------------------------
tench 0.6111934781074524
chain saw 0.344500333070755
English springer 0.011400890536606312
gas pump 0.009950699284672737
French horn 0.007910788990557194
--------------------------------
Detecting for class test_chainsawtrain.jpg model mycnn2
--------------------------------
chain saw 0.6810593008995056
English springer 0.23049525916576385
gas pump 0.05660108104348183
French horn 0.02358389087021351
garbage truck 0.0029852665029466152
--------------------------------
Detecting for class test_frenchhorn.jpg model mycnn2
--------------------------------
French horn 0.9243784546852112
chain saw 0.02931187115609646
tench 0.01774682104587555
church 0.01660805754363537
golf ball 0.006724261678755283
--------------------------------
Detecting for class test_frenchhorntrain.jpg model mycnn2
--------------------------------
French horn 0.7874661087989807
chain saw 0.15104267001152039
church 0.03641435503959656
gas pump 0.014874609187245369
golf ball 0.0059154825285077095
--------------------------------
Detecting for class test-golfball.jpg model mycnn2
--------------------------------
golf ball 0.7764129638671875
English springer 0.13179312646389008
chain saw 0.04314707592129707
gas pump 0.01596427708864212
tench 0.011418420821428299
--------------------------------
```
