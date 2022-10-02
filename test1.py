from scipy import stats
from scipy.stats import norm
from numpy.random import default_rng

rng = default_rng()


def main():
    ran = norm.rvs(size=100, random_state=rng)
    print(ran)


if __name__ == "__main__":
    main()
