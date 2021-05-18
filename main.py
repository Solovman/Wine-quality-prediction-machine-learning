from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import scale

input_file = 'wine.csv'
data_set = read_csv(input_file, delimiter=',')
print(data_set.head(7))

# properties of wine
x = data_set.values[::, 1:14]
# type of wine
y = data_set.values[::, 0:1].ravel()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7)


clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))

x_train_draw = scale(x_train[::, 0:2])
x_test_draw = scale(x_test[::, 0:2])

clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
clf.fit(x_train_draw, y_train)

x_min, x_max = x_train_draw[:, 0].min() - 1, x_train_draw[:, 0].max() + 1
y_min, y_max = x_train_draw[:, 1].min() - 1, x_train_draw[:, 1].max() + 1

h = 0.02

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
pred = pred.reshape(xx.shape)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.figure()
plt.pcolormesh(xx, yy, pred, cmap=cmap_light)
plt.scatter(x_train_draw[:, 0], x_train_draw[:, 1],
            c=y_train, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.title("Score: %.0f percents" % (clf.score(x_test_draw, y_test) * 100))
plt.show()
