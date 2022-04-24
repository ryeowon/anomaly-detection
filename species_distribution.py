# Authors: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Jake Vanderplas <vanderplas@astro.washington.edu>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import Bunch
from sklearn.datasets import fetch_species_distributions
from sklearn import svm, metrics

try:
    from mpl_toolkits.basemap import Basemap

    basemap = True
except ImportError:
    basemap = False


def construct_grids(batch):
    xmin = batch.x_left_lower_corner + batch.grid_size
    xmax = xmin + (batch.Nx * batch.grid_size)
    ymin = batch.y_left_lower_corner + batch.grid_size
    ymax = ymin + (batch.Ny * batch.grid_size)

    xgrid = np.arange(xmin, xmax, batch.grid_size)
    ygrid = np.arange(ymin, ymax, batch.grid_size)
    #print(xmin, xmax, ymin, ymax)
    #print(batch.grid_size)
    #print(xgrid, ygrid)

    return (xgrid, ygrid)

def create_species_bunch(species_name, train, test, coverages, xgrid, ygrid):
    bunch = Bunch(name=" ".join(species_name.split("_")[:2]))
    #print(bunch)
    species_name = species_name.encode("ascii")
    points = dict(test=test, train=train)
    #print(species_name)
    for label, pts in points.items():
        #print(pts)
        pts = pts[pts["species"] == species_name]
        #print(pts)
        bunch["pts_%s" % label] = pts

        ix = np.searchsorted(xgrid, pts["dd long"]) # longitude
        iy = np.searchsorted(ygrid, pts["dd lat"])  # latitude
        bunch["cov_%s" % label] = coverages[:, -iy, ix].T

    #print (bunch)
    return bunch

def plot_species_distribution(
    species=("bradypus_variegatus_0", "microryzomys_minutus_0")
):
    if len(species) > 2:
        print(
            "Note: when more than two species are provided,"
            " only the first two will be used"
        )

    data = fetch_species_distributions()

    xgrid, ygrid = construct_grids(data)
    X, Y = np.meshgrid(xgrid, ygrid[::-1])

    BV_bunch = create_species_bunch(species[0], data.train, data.test, data.coverages, xgrid, ygrid)
    MM_bunch = create_species_bunch(species[1], data.train, data.test, data.coverages, xgrid, ygrid)

    # background points
    np.random.seed(13)
    background_points = np.c_[
        np.random.randint(low=0, high=data.Ny, size=10000),
        np.random.randint(low=0, high=data.Nx, size=10000),
    ].T

    land_reference = data.coverages[6] #coverage: georelational data model that stores vector data

    for i, species in enumerate([BV_bunch, MM_bunch]):
        print("_" * 80)
        print("Modeling distribution of species '%s'" % species.name)

        mean = species.cov_train.mean(axis=0)
        std = species.cov_train.std(axis=0)
        train_cover_std = (species.cov_train - mean) / std # 표준화

        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.5)
        clf.fit(train_cover_std)

        plt.subplot(1, 2, i + 1)

        # 남아메리카 지도 그리기
        if basemap:
            print(" - plot coastlines using basemap")
            m = Basemap(
                projection="cyl",
                llcrnrlat=Y.min(),
                urcrnrlat=Y.max(),
                llcrnrlon=X.min(),
                urcrnrlon=X.max(),
                resolution="c",
            )
            m.drawcoastlines()
            m.drawcountries()
        else:
            print(" - plot coastlines from coverage")
            plt.contour(
                X, Y, land_reference, levels=[-9998], colors="k", linestyles="solid"
            )
            plt.xticks([])
            plt.yticks([])
        
        Z = np.ones((data.Ny, data.Nx), dtype=np.float64)

        idx = np.where(land_reference > -9999)
        #print(idx)
        coverages_land = data.coverages[:, idx[0], idx[1]].T
        #print(coverages_land)
        #print(mean)
        #print(Z)
        pred = clf.decision_function((coverages_land - mean) / std)
        
        Z *= pred.min()
        #print(Z)
        Z[idx[0], idx[1]] = pred
        #print(Z[idx[0], idx[1]])

        levels = np.linspace(Z.min(), Z.max(), 25)

        plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Reds)
        plt.colorbar(format="%.2f")

        plt.scatter(
            species.pts_train["dd long"],
            species.pts_train["dd lat"],
            s=10,
            c="black",
            marker="^",
            label="train",
        )
        plt.scatter(
            species.pts_test["dd long"],
            species.pts_test["dd lat"],
            s=10,
            c="yellow",
            label="test",
            edgecolors="k",
        )
        plt.legend()
        plt.title(species.name)
        plt.axis("equal")
        
        # auc 계산
        pred_background = Z[background_points[0], background_points[1]]
        pred_test = clf.decision_function((species.cov_test - mean) / std)
        #print(pred_test)
        scores = np.r_[pred_test, pred_background]
        y = np.r_[np.ones(pred_test.shape), np.zeros(pred_background.shape)]
        fpr, tpr, thresholds = metrics.roc_curve(y, scores)
        #print(thresholds)
        roc_auc = metrics.auc(fpr, tpr)
        plt.text(-35, -70, "AUC: %.3f" % roc_auc, ha="right")
        print("\n Area under the ROC curve : %f" % roc_auc)


plot_species_distribution()
plt.show()