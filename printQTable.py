def printQTable(tableName):
  columnIndex = [0,1,2,3,4,5,6,7,8,9]
  print("\033[1m        0      1        2      3       4       5        6      7       8       9\n"+'\033[0m')
  for j in range(0, 10, 2):
      # print(re.sub(r' *\n *', '\n', np.array_str(np.c_[qtable[i, j], qtable[i+1, j]]).replace('[', '').replace(']', '').strip()), " | ",re.sub(r' *\n *', '\n', np.array_str(np.c_[qtable[i+2, j], qtable[i+3, j]]).replace('[', '').replace(']', '').strip()), " | ",np.array_str(np.c_[qtable[i+4, j], qtable[i+5, j]]).replace('[', '').replace(']', '').strip()), " | ",np.array_str(np.c_[qtable[i+6, j], qtable[i+7, j]]).replace('[', '').replace(']', '').strip()), " | ",np.array_str(np.c_[qtable[i+8, j], qtable[i+9, j]]).replace('[', '').replace(']', '').strip()), " | ",np.array_str(np.c_[qtable[i+10, j], qtable[i+11, j]]).replace('[', '').replace(']', '').strip()), " | ",np.array_str(np.c_[qtable[i+12, j], qtable[i+13, j]]).replace('[', '').replace(']', '').strip()), " | ",np.array_str(np.c_[qtable[i+14, j], qtable[i+15, j]]).replace('[', '').replace(']', '').strip()), " | ",)
      print("\033[1m%s  \033[0m   %.3f  %.3f | %.3f  %.3f | %.3f  %.3f | %.3f  %.3f | %.3f  %.3f " % (columnIndex[j], tableName[j, 0], tableName[j, 1], tableName[j, 2], tableName[j, 3], tableName[j, 4], tableName[j, 5], tableName[j, 6], tableName[j, 7], tableName[j, 8], tableName[j, 9]))
      print("\033[1m%s  \033[0m   %.3f  %.3f | %.3f  %.3f | %.3f  %.3f | %.3f  %.3f | %.3f  %.3f " % (columnIndex[j+1], tableName[j+1, 0], tableName[j+1, 1], tableName[j+1, 2], tableName[j+1, 3], tableName[j+1, 4], tableName[j+1, 5], tableName[j+1, 6], tableName[j+1, 7], tableName[j+1, 8], tableName[j+1, 9]))
      if(j !=8 ):
        print("     ---------------------------------------------------------------------------")
