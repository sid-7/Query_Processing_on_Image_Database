# Query_Processing_on_Image_Database
Feature Extraction, Dimensionality Reduction and Query Processing on an Image Database.

The project uses dataset associated with the following publication:<br>
             <i>Mahmoud Afifi. “11K Hands: Gender recognition and biometric identification using a large dataset of hand images.” M. Multimed Tools Appl (2019) 78: 20835</i>

# Phase-1:
- Feature-Extraction:
  - Extracting features using from the 11k hand images using the following method:
    - Color Moments
    - Local Binary Pattern
    - Histograms of oriented gradients
    - Scale-invariant feature transform
  - Implemented a program to extract k similar images, provided a image-id using the extracted feature descriptors and using Cosine Similarity as a distance measure.
    
# Phase-2:
 - Impletented a program to identify latent semantics using the following 4 methods:
    - Pricipal Component Analysis (PCA)
    - Singular Value Decomposition (SVD)
    - Non-negative Matrix Factorization (NMF)
    - Latent Dirichlet Analysis (LDA)
 - Implemented a program to list most related <i>m</i> images in the <i>k</i> dimanesional latent space.
 - Implemented a program which (a) lets the user to chose among one of the four feature models and (b) given one of the labels
    - left-hand
    - right-hand
    - dorsal
    - palmar
    - with accessories
    - without accessories
    - male
    - female<br>
identifies (and lists) k latent semantics for images with the corresponding metadata using (c) one of the following techniques chosen by the user
- Implement a program which given ( a subject ID, identifies and visualizes the most related 3 subjects).

 # Phase-3
  - Implemented a program which, given a folder with dorsal/palmar labeled images and for a user supplied c,
    - computes c clusters associated with dorsal-hand images (visualize the resulting image clusters),
    - computes c clusters associated with palmar-hand images (visualize the resulting image clusters),<br>
and, given a folder with unlabeled images, the system labels them as <b> dorsal-hand vs palmar-hand </b> using descriptors of these clusters.
  - Implemented Personalised Page Rank (PPR) Algorithm to detect and visualise K most dominant images for a given image-id.
  - Implemented SVM, Decision-tree and PPR based Algorithm to classify a set of images as <b> dorsal-hand vs palmar-hand </b>.
  - Implemented Locality Sensitive Hashing based search algorithm to visualise most similar images for a given image-id.
  - Impleted a Relevant Feedback System.
