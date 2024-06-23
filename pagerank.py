import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    n = len(corpus)
    distribution = dict.fromkeys(corpus.keys(), (1 - damping_factor) / n)
    
    links = corpus[page]
    if links:
        link_prob = damping_factor / len(links)
        for link in links:
            distribution[link] += link_prob
    else:
        # If no links, assume equal probability to all pages including itself
        for key in distribution.keys():
            distribution[key] += damping_factor / n

    return distribution



def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = dict.fromkeys(corpus.keys(), 0)
    sample = random.choice(list(corpus.keys()))

    for _ in range(n):
        pagerank[sample] += 1
        distribution = transition_model(corpus, sample, damping_factor)
        sample = random.choices(list(distribution.keys()), weights=distribution.values(), k=1)[0]

    for page in pagerank:
        pagerank[page] /= n

    return pagerank



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    n = len(corpus)
    pagerank = dict.fromkeys(corpus.keys(), 1 / n)
    new_rank = dict.fromkeys(corpus.keys(), 0)
    convergence_threshold = 0.001

    while True:
        for page in pagerank:
            total = (1 - damping_factor) / n
            for potential_linker in corpus:
                if page in corpus[potential_linker] or len(corpus[potential_linker]) == 0:
                    link_count = len(corpus[potential_linker]) if len(corpus[potential_linker]) != 0 else n
                    total += damping_factor * pagerank[potential_linker] / link_count
            new_rank[page] = total

        if all(abs(new_rank[page] - pagerank[page]) < convergence_threshold for page in pagerank):
            break

        pagerank = new_rank.copy()

    return pagerank



if __name__ == "__main__":
    main()
