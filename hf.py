import numpy as np
from numpy import linalg

#First, we read the file "vnn" to obtain the nuclear repulsion energy

readfile = open('vnn','r')
sepfile = readfile.read().split('\n')
readfile.close()
Nee = 10
for pair in sepfile:
	nuc_rep = float(pair)
	break
#print('The nuclear repulsion energy is ',nuc_rep,'Hartree')

# Second, we read the file "one-electron" to obtain the one-electron integral matrix
nbf = 0
readfile = open('one-electron','r')
sepfile = readfile.read().split('\n')
readfile.close()
one = []
for pair in sepfile:
	if pair == '':
		break
	else:
		matrix = pair.split()
		nbf = nbf+1
		one.append(matrix)

for i in range(nbf):
	for j in range(nbf):
		one[i][j] = float(one[i][j])
#print('The one electron integral= \n',one)

# Similarly, we read the file "overlap" to obtain the overlap integral matrix
readfile = open('overlap','r')
sepfile = readfile.read().split('\n')
readfile.close()
S = np.zeros((nbf,nbf))
count = 0
for pair in sepfile:
	if pair == '':
		break
	else:
		matrix = pair.split()
		for i in range(nbf):
			S[count][i] = float(matrix[i])
		count = count + 1
#print('The overlap matrix= \n',S)

#It is quite tricky for reading two-electron integral matrix, as it is a four dimensional array.
readfile = open('two-electron','r')
sepfile = readfile.read().split('\n')
readfile.close()
two = []
two_ele = np.zeros((nbf,nbf,nbf,nbf))
line =1 
sum_int = 0
for pair in sepfile:
	if pair == '':
		break
	else:
		matrix = pair.split()
		mi = int(matrix[0])
		v = int(matrix[1])
		rho = int(matrix[2])
		sigma = int(matrix[3])
		int_2 = float(matrix[4])
		two_ele[mi][v][rho][sigma] = int_2
		two_ele[v][mi][rho][sigma] = two_ele[mi][v][rho][sigma]
		#swapping the positions of wavefunctions within 'bra' and 'ket' does not make any difference.
		two_ele[mi][v][sigma][rho] = int_2
		two_ele[v][mi][sigma][rho] = int_2
		two_ele[rho][sigma][mi][v] = int_2
		two_ele[rho][sigma][v][mi] = int_2
		two_ele[sigma][rho][mi][v] = int_2
		two_ele[sigma][rho][v][mi] = int_2
		#sum_int = sum_int+abs(int_2)
#print(sum_int)

#A summation procedure to check if the two-electron integral matrix is not read correctly.
sum_int = 0
sum_int_sq = 0
for i in range(nbf):
	for j in range(nbf):
		for k in range(nbf):
			for p in range(nbf):
				sum_int = sum_int + abs(two_ele[i][j][k][p])
				sum_int_sq = sum_int_sq + two_ele[i][j][k][p]**2

#print('The sum of two electron integral= \n',sum_int)
#print('The sum of square of two electron integral= \n',sum_int_sq)

#Now, we need to diagonalize the overlap matrix, S. 
#We can do that by firstly determine the eigenvalues of S.
#And then, make a matrix with a diagonal s^0.5
#X is the diagonal matrix of S.
s, L = np.linalg.eigh(S)
#print('s = ',s)
s2 = [si**(-0.5) for si in s]
s3 = np.zeros((nbf,nbf))
for i in range(nbf):
	for j in range(nbf):
		if i == j:
			s3[i][j] = s2[i]
		else:
			pass
tran_L = np.transpose(L)
X = np.mat(L)*np.mat(s3)*np.mat(tran_L)
#print(X)

#Calculation of D and G matrices, which are useful for updating the coefficients in the linear combination of basis sets.
D = np.zeros((nbf,nbf))
G = np.zeros((nbf,nbf))
for i in range(nbf):
	for j in range(nbf):
		for k in range(nbf):
			for p in range(nbf):
				G[i][j] = G[i][j]+D[k][p]*(2*two_ele[i][j][k][p]-two_ele[i][k][j][p])

#print(G)

#Now, we are in position to implement the self-consistency of HF method
#In fact, from the procedures above, we have estimated the value of the Fock operator using initial guess
#With the Fock operator evaluated using initial guess, we can calculate the new values of coefficients, which could
#be used as initial guess in the next iteration

iteration_count = 0
err = 1
E_hf0 = 0

while err > 10**(-10):
	sum_h = 0
	F = one+G #update the Fock operator
	for i in range(nbf):
		for j in range(nbf):
			sum_h = sum_h+D[i][j]*(one[i][j]+F[i][j])
	E_hf = nuc_rep + sum_h
	#print(E_hf)
	F_prim = np.mat(X)*np.mat(F)*np.mat(X)
	elipson, c_prim = np.linalg.eigh(F_prim)
	#print(elipson)
	C = np.mat(X)*np.mat(c_prim)
	c = np.zeros((nbf,nbf))
	for i in range(nbf):
		row = C[i,:].reshape(1,nbf).T
		for j in range(nbf):
			c[i][j] = row[j]
	D = np.zeros((nbf,nbf))
	for i in range(nbf):
		for j in range(nbf):
			for k in range(int(Nee/2)):
				D[i][j] = D[i][j]+c[i][k]*c[j][k]
	G = np.zeros((nbf,nbf))
	for i in range(nbf):
		for j in range(nbf):
			for k in range(nbf):
				for p in range(nbf):
					G[i][j] = G[i][j]+D[k][p]*(2*two_ele[i][j][k][p]-two_ele[i][k][j][p])
	err =abs(E_hf0-E_hf)
	print("Total energy is ",E_hf,"Hartree and iteration count is ",iteration_count)
	E_hf0 = E_hf
	iteration_count = iteration_count + 1
	#if iteration_count == 10:
	#	break
	#else:
	#	pass

