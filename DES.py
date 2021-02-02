import numpy as np
import time
import string
# initial permutation input 64 – output 64
IP = [58, 50, 42, 34, 26, 18, 10, 2,
      60, 52, 44, 36, 28, 20, 12, 4,
      62, 54, 46, 38, 30, 22, 14, 6,
      64, 56, 48, 40, 32, 24, 16, 8,
      57, 49, 41, 33, 25, 17, 9, 1,
      59, 51, 43, 35, 27, 19, 11, 3,
      61, 53, 45, 37, 29, 21, 13, 5,
      63, 55, 47, 39, 31, 23, 15, 7]

#inverse initial permutation
FP = [40, 8, 48, 16, 56, 24, 64, 32,
        39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30,
        37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28,
        35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26,
        33, 1, 41, 9, 49, 17, 57, 25]

#expansion permutation input 32 – output 48
EBox = [32,1,2,3,4,5,
            4,5,6,7,8,9,
            8,9,10,11,12,13,
            12,13,14,15,16,17,
            16,17,18,19,20,21,
            20,21,22,23,24,25,
            24,25,26,27,28,29,
            28,29,30,31,32,1]

SBox =[
		# S1
		[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
		 0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
		 4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
		 15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],

		# S2
		[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10,
		 3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
		 0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15,
		 13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],

		# S3
		[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8,
		 13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,
		 13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7,
		 1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],

		# S4
		[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15,
		 13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,
		 10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4,
		 3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],

		# S5
		[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9,
		 14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,
		 4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14,
		 11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],

		# S6
		[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11,
		 10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8,
		 9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6,
		 4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],

		# S7
		[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1,
		 13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,
		 1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2,
		 6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],

		# S8
		[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7,
		 1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,
		 7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8,
		 2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
	]

#Permutation table : input 32 – output 32
F_PBox = [16, 7, 20, 21, 29, 12, 28, 17,
              1, 15, 23, 26, 5, 18, 31, 10,
              2, 8, 24, 14, 32, 27, 3, 9,
              19, 13, 30, 6, 22, 11, 4, 25 ]
#Permuted choice 2 : input 56 – output 48
key_PBox = [14,    17,   11,    24,     1,    5,
                  3,    28,   15,     6,    21,   10,
                 23,    19,   12,     4,    26,    8,
                 16,     7,   27,    20,    13,    2,
                 41,    52,   31,    37,    47,   55,
                 30,    40,   51,    45,    33,   48,
                 44,    49,   39,    56,    34,  53,
                 46,    42,   50,    36,    29,   32]

#Permuted choice 1 : input 64 – output 56
pc1=[57,49,41,33,25,17,9,
  1,58,50,42,34,26,18,
  10,2,59,51,43,35,27,
  19,11,3,60,52,44,36,
  63,55,47,39,31,23,15,
  7,62,54,46,38,30,22,
  14,6,61,53,45,37,29,
  21,13,5,28,20,12,4
	]

#Left Circular Shift : input 56 – output 56
__left_rotations = [
1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]


def E_box(right):
    """
    expansion permutation using EBox table.
    Args:
        right: array of len 32 binary.
    Returns:
        expanded: array of len 48 binary.
    """     
    expanded = np.empty(48)
    j = 0
    for i in EBox:
        expanded[j] = right[i - 1]
        j += 1
    expanded = list(map(int,expanded))
    expanded = np.array(expanded)
    return expanded

def xor(right,roundkey):
    """
    Xor the output of expantion with round key.
    Args:
        right:  array of len 48 binary.
        roundkey: array of len 48 binary.
    Returns:
        xorresult: array of len 48 binary.
    """   
    xorresult = np.logical_xor(right,roundkey)
    xorresult  = xorresult.astype(int)
    return xorresult
    
def sboxloopup(sinput,x):
    tableno = x - 1
    row = int((np.array2string(sinput[0]) + np.array2string(sinput[5])),2)
    column = sinput[1:5]
    column = np.array2string(column)
    column = column[1:8].replace(" ", "")
    column = int(column,2)
    

    elementno = (16 * row) + column
    soutput = SBox[tableno][elementno]
    soutput = list(np.binary_repr(soutput, width=4))
    #converting to list twice seems redundant but seems to be the only simple way as map always returns map object
    soutput= np.array(list(map(int, soutput)))
    return soutput

def sbox(sboxin):
    """
    Substitution using SBox table.
    Args:
        sboxin: array of len 48 binary.
    Returns:
        sboxout: array of len 32 binary.
    """     
    #takes 48 bit input and return 32 bit
    sboxin1 = sboxin[0:6]
    sboxout1 = sboxloopup(sboxin1,1)
    sboxin2 = sboxin[6:12]
    sboxout2 = sboxloopup(sboxin2,2)
    sboxin3 = sboxin[12:18]
    sboxout3 = sboxloopup(sboxin3, 3)
    sboxin4 = sboxin[18:24]
    sboxout4 = sboxloopup(sboxin4, 4)
    sboxin5 = sboxin[24:30]
    sboxout5 = sboxloopup(sboxin5, 5)
    sboxin6 = sboxin[30:36]
    sboxout6 = sboxloopup(sboxin6, 6)
    sboxin7 = sboxin[36:42]
    sboxout7 = sboxloopup(sboxin7, 7)
    sboxin8 = sboxin[42:48]
    sboxout8 = sboxloopup(sboxin8, 8)
    sboxout = np.concatenate([sboxout1,sboxout2,sboxout3,sboxout4,sboxout5,sboxout6,sboxout7,sboxout8])
    return sboxout

def f_permute(topermute):
    """
    round final permutation using F_PBox table.
    Args:
        topermute: array of len 32 binary.
    Returns:
        permuted: array of len 32 binary.
    """     
    permuted= np.empty(32)
    j = 0
    for i in F_PBox:
        permuted[j] = topermute[i - 1]
        j += 1
    return permuted

def f_function(right,roundkey):
    """
    Args:
        right:     array of len 32 binary.
        roundkey : array of len 48 binary.
    Returns:
        xorstream: array of len 32 binary.
    """     
    expanded = E_box(right)
    xored = xor(expanded,roundkey)
    sboxed = sbox(xored)
    xorstream = f_permute(sboxed)
    return xorstream

def round(data,roundkey):
    """
    Encryption of one round.
    Args:
        data: array of len 64 binary.
        roundkey : array of len 48 binary.
    Returns:
        roundOut: array of len 64 binary.
    """     
    l0 = data[0:32]
    r0 = data[32:64]
    xorstream = f_function(r0,roundkey)
    r1 = xor(l0,xorstream)
    l1 = r0
    roundOut = np.empty_like(data)
    roundOut[0:32]  = l1
    roundOut[32:64] = r1
    return(roundOut)

def initial_final_permutation(data,x):
    """
    Initial and final permutation using IP and FP tables respectively.
    Args:
        data: array of len 64 binary.
        x : integer ( 0 for initial permutation other wise for final permutation).
    Returns:
        permute1: array of len 64 binary in case of x = 0. 
        permute2: array of len 64 binary in case of x != 0. 
    """     
    permute1 = np.empty_like(IP)
    if x == 0:
        j = 0
        for i in IP:
            permute1[j] = data[i-1]
            j += 1
        return(permute1)
    else:
        permute2 = np.empty_like(FP)
        k = 0
        for l in FP:
            permute2[k] = data[l-1]
            k += 1
        return(permute2)

def key_permutation_choice1(key):
    """
    permutation Choice 1 using pc1 table which takes key of 64 bits and returns it with 56 bits.
    Args:
        key: array of len 64 binary.
    Returns:
        permute1: array of len 56 binary. 
    """     
    permute1 = np.empty_like(pc1)
    for i in range(56):
        permute1[i]=key[pc1[i]-1]
    
    return(permute1)

def halfkey_left_circular_shift(toshift,n):
    """
    rotate left the half key based on the round number.
    Args:
        toshift: array of len 28 binary .
        n: integer represent round number.
    Returns:
        toshift: array of len 28 binary . 
    """     
    
    if (n == 1) or (n == 2) or (n == 9) or (n == 16):
        toshift= np.roll(toshift,-1)
        return toshift
    else:
        toshift = np.roll(toshift, -2)
        return toshift

def key_permutation_choice2(key16):
    """
    Permute the 16 keys using key_PBox table .
    Args:
        key16: 16*65 array of lists .
    Returns:
        keypermuted: 16*84 array of lists. 
    """   
    keypermuted = np.empty([16,48])
    l = 0
    for k in key16:
        j = 0
        for i in key_PBox:
            keypermuted[l][j] = k[i - 1]
            j += 1
        l += 1
    return keypermuted

def Key_Generator(key):
    """
    Generate 16 keys of len 48 bits.
    Args:
        key: array of len 64 binary.
    Returns:
        key16: 16*48 array of lists. 
    """ 
    #permutation choice 1
    key=key_permutation_choice1(key)
    
    left = key[0:28]
    right = key[28:56]
    shifted = np.zeros(56)
    key16 = np.zeros([16,56])
    
    #generate 16 keys for 16 rounds
    #key circular shift 
    for i in range(1,17):
        shifted[0:28] =  halfkey_left_circular_shift(left,i)
        shifted[28:56] = halfkey_left_circular_shift(right,i)
        left = shifted[0:28]
        right = shifted[28:56]

    #add shifted to key16 and return key16
        key16[i - 1] = shifted
    
    #permutation choice 2
    key16 = key_permutation_choice2(key16)
    key16 = [list(map(int, x)) for x in key16]
    key16 = np.array(key16)
    return key16

# Binary to Hex Conversion 
def bin2hex(s): 
    mp = {"0000" : '0',  
          "0001" : '1', 
          "0010" : '2',  
          "0011" : '3', 
          "0100" : '4', 
          "0101" : '5',  
          "0110" : '6', 
          "0111" : '7',  
          "1000" : '8', 
          "1001" : '9',  
          "1010" : 'A', 
          "1011" : 'B',  
          "1100" : 'C', 
          "1101" : 'D',  
          "1110" : 'E', 
          "1111" : 'F' } 
    hex = "" 
    for i in range(0,len(s),4): 
        ch = "" 
        ch = ch + s[i] 
        ch = ch + s[i + 1]  
        ch = ch + s[i + 2]  
        ch = ch + s[i + 3]  
        hex = hex + mp[ch] 
          
    return hex

# Hex to Binary Conversion 
def hex2bin(s): 
    mp = {'0' : "0000",  
          '1' : "0001", 
          '2' : "0010",  
          '3' : "0011", 
          '4' : "0100", 
          '5' : "0101",  
          '6' : "0110", 
          '7' : "0111",  
          '8' : "1000", 
          '9' : "1001",  
          'A' : "1010", 
          'B' : "1011",  
          'C' : "1100", 
          'D' : "1101",  
          'E' : "1110", 
          'F' : "1111",
          'a' : "1010", 
          'b' : "1011",  
          'c' : "1100", 
          'd' : "1101",  
          'e' : "1110", 
          'f' : "1111"} 
    bin = "" 
    for i in range(len(s)): 
        bin = bin + mp[s[i]] 
    return bin

def DES_Encryption(data_array,key16,no_of_encryption):  
    """
    Encrypt the block using key16 from key 0 to key 15.
    Args:
        data_array: array of len 64 binary to encrypt.
        key16: 16*48 array of lists.
        no_of_encryption: integer
    Returns:
        data_string: string of len 16 represent encrypted block. 
    """ 
    for k in range(int(no_of_encryption)):
    
        #initial permutation
        data_array = initial_final_permutation(data_array,0)
        
        #16 rounds
        for i in range(16):
            data_array = round(data_array,key16[i])

        #32-bit swap
        data_array = np.roll(data_array,32)
        
        #inverse of initail permutation
        data_array = (initial_final_permutation(data_array, 1))
    
    #convert from array to string
    data_string=""
    data_string = np.array_str(data_array)
    data_string= data_string.replace(" ","")
    data_string = data_string.replace('[','')
    data_string = data_string.replace(']','')
    data_string = data_string.replace('\n','')
    #convert binary into hexa
    data_string=bin2hex(data_string)
            
    return data_string

def DES_Decryption(data_array,key16):
    """
    Encrypt the block using key16 from key 15 to key 0.
    Args:
        data_array: array of len 64 binary to encrypt.
        key16: 16*48 array of lists.
    Returns:
        data_string: string of len 16 represent decrypted block. 
    """ 
    #initial permutation
    data_array = initial_final_permutation(data_array, 0)
    #16 rounds
    for i in range(16):
        data_array = round(data_array, key16[16 - (i + 1)])
    #32-bit swap
    data_array = np.roll(data_array, 32)
    #inverse of initail permutation
    data_array = (initial_final_permutation(data_array, 1))
    
    #convert from array to string
    data_string=""
    data_string = np.array_str(data_array)
    data_string= data_string.replace(" ","")
    data_string = data_string.replace('[','')
    data_string = data_string.replace(']','')
    data_string = data_string.replace('\n','')
    #convert binary into hex
    data_string=bin2hex(data_string)
    return data_string

def main():
    print("                                        Welcome_to_DES_Cipher                                       \n\n")
    
    while(1):
        print("Enter 1 for Encryption or 2 for Decryption or 0 for Exit: ")
        sel = input()
        #Encryption
        if sel == '1':
            print("Enter The Key Of 16 Hex Characters for DES Cipher: ")
            key=input()
            print("Enter The PlainText Of 16 Hex Characters: ")
            data=input()
            print("Enter the number of times for encryption : ")
            no_of_encryption=input()
            #Convert hex into binary
            key=hex2bin(key)
            data=hex2bin(data)
            #conver string into array of character
            key_array=[]
            data_array=[]
            for i in range(64):
                key_array.append(key[i])

            for i in range(64):
                data_array.append(data[i])
                
            #key generation
            key16= Key_Generator(key_array) 
        
            print("The Encrypted Data is : " + DES_Encryption(data_array,key16,no_of_encryption))
            continue
            
        #Decryption    
        elif sel == '2':
            print("Enter The Key Of 16 Hex Characters for DES Cipher: ")
            key=input()
            print("Enter The PlainText Of 16 Hex Characters: ")
            data=input()
            #Convert hex into binary
            key=hex2bin(key)
            data=hex2bin(data)
            #conver string into array of character
            key_array=[]
            data_array=[]
            for i in range(64):
                key_array.append(key[i])

            for i in range(64):
                data_array.append(data[i])
                
            #key generation
            key16= Key_Generator(key_array) 
            print("The Decrypted Data is : " + DES_Decryption(data_array,key16))
            continue
            
        #Exit 
        elif sel == '0':
            break;
        else:
            print("Invalid Operation: please Enter a Correct Number\n\n")
            continue
   
    
        



main()