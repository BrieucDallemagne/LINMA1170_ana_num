import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def extract_color(database, color) :
    S  = database[f"S_{color}"]
    U_str = np.array(database[f"U_{color}"])
    U = list()
    for u_str in U_str :
        u = list()
        u_str = u_str[1:-1].split(' ')
        for val in u_str :
            u.append(float(val))
        U.append(u)
    U = np.array(U)
    print(U)
    Vh = database[f"Vh_{color}"]
    print(np.shape(S))
    print(U)
    print(np.shape(U))
    print(np.shape(Vh))
    return U @ np.diag(S) @ Vh

def show_image(database) :
    blue_data = np.clip(extract_color(database, 'blue'), 0, 255).astype(np.uint8)
    green_data = np.clip(extract_color(database, 'green'), 0, 255).astype(np.uint8)
    red_data = np.clip(extract_color(database, 'red'), 0, 255).astype(np.uint8)
    
    compressed_image = np.dstack((red_data, green_data, blue_data))
    plt.imshow(compressed_image)
    plt.show()


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description="Restitution d'image")
    parser.add_argument('-f', '--file', type=str, help="Fichier avec les données de l'image")
    args = parser.parse_args()
    show_image(pd.read_csv(args.file))