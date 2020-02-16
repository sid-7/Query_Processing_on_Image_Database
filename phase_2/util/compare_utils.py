from phase_1.sandeepkunichi.driver import *

hog_df = feature_storage.load_to_df(ModelType.COLOR_MOMENTS)

hist_1 = hog_df.loc[2]
hist_2 = hog_df.loc[3]
hist_3 = hog_df.loc[4]

print hist_1.tolist
print hist_2.tolist


def dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


print dist(hist_1, hist_2)
