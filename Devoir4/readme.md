# Projet de LINMA1170
Dans le cadre du cours LINMA1170 nous avons réalisé 2 applications de la décomposition en valeurs singulières (SVD)  : une application de compression d'image de débruitage d'image.

# Run the code
Pour run le code, il suffit de lancer la commande suivante :
```bash
pip install -r requirements.txt
```

et ensuite vous pouvez run le fichier python de votre choix, il en existe 4:
- `compression_image.py [-h] [-f FILE] [-o OUTPUT_FILE] [-k COMPONENTS]` (pour la compression d'image)
- `comparaison_audio.py` (pour la comparaison de deux fichiers audio avec des valeurs singulières différentes)
- `débruitage_audio.py [-h] [-f FILE] [-o OUTPUT_FILE] [-k COMPONENTS]` (pour le débruitage d'un fichier audio)
- `plot_erreur.py` (pour plot l'erreur de compression d'image en fonction du nombre de valeurs singulières gardées)