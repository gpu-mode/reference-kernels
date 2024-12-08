########
# Evaluation scripts to run for leaderboard results
########

py_eval = """
from reference import metric

def main():
    s = metric()
    print(f'score:{s}')

if __name__ == '__main__':
    main()

"""

cu_eval = """

"""
