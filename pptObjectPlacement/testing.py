l1 = [[0, 2, 5, 0, 0, 0],
      [3, 4, 6, 0, 0, 0],
      [0, 1, 1, 1, 1, 0],
      [0, 0, 1, 1, 1, 0],
      [0, 0, 1, 1, 1, 0],
      [0, 0, 0, 0, 0, 0]]

for w in range(0, 6):
    for h in range(0, 4):
        print(f"----\ncheck[{h},{w}]:{l1[h][w]}")
        #if not l1[h][w] == 1:
        #    w+=1
        #    break
        #else:
        #    break
        #    w+=1
        #print(f"changed to: [{h},{w}]:{l1[h][w]}\n----")
    #w+=1
print(f"w:{w}, h:{h}")


[255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255
 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255
 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255]