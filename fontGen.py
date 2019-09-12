import random
file_data = open("test.csv","w")
def data():
    for I in range(10000): 
        R = random.randint(0,255)
        G = random.randint(0,255)
        B = random.randint(0,255)
        color = [R,G,B]
        lumin = R*0.299 + G *.587+ B*.114
        if lumin > 186:
            file_data.write(str(color[0])+"," + str(color[1])+ "," +str(color[2])+"\n"+str(0))
        elif lumin < 186:
            file_data.write(str(color[0])+"," + str(color[1])+ "," +str(color[2])+"\n"+str(1))

data()