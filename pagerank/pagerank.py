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
    dfp = 1 - damping_factor
    pd = {}
    if corpus[page]:
        for i in corpus[page]:
            pd[i] = damping_factor / len(corpus[page])

        for i in corpus:
            pd[i] = pd.get(i,0) + (dfp / len(corpus))
    else:
        for i in corpus:
            pd[i] = pd.get(i,0) + (1 / len(corpus))
    return pd
    


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pd = transition_model(corpus,random.choice(list(corpus.keys())),DAMPING)
    pagerank = {}
    for i in range(n):
        sample = random.choices(list(pd.keys()),weights = list(pd.values()))[0]
        pagerank[sample] = pagerank.get(sample,0) + 1
        pd = transition_model(corpus,sample, DAMPING)
    for key in pagerank:
        pagerank[key] = pagerank[key] / n
    return pagerank

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    dfp = 1 - damping_factor
    inv_corpus = {page : [] for page in corpus}
    pagerank = {}
    con_rate = float("inf")
    for page in corpus:
        pagerank[page] = damping_factor / len(corpus)
        if corpus[page]:
            for link in corpus[page]:
                inv_corpus[link].append(page)
        else:
            for link in corpus:
                inv_corpus[link].append(page)
    
    while con_rate > 0.0001:
        next_rank = {}
        for page in pagerank:
            
            for linked_page in inv_corpus[page]:
                divisor = len(corpus[linked_page]) if corpus[linked_page] else len(corpus)
                next_rank[page] = (next_rank.get(page,0)+(pagerank[linked_page] / divisor))
            
            next_rank[page] = (next_rank.get(page,0)*.85) + dfp/len(corpus)
            
                
        con_vals = [pagerank[x]-next_rank[x] if pagerank[x]-next_rank[x] > 0 else (pagerank[x]-next_rank[x])*-1 for x in pagerank.keys()]
        con_rate = max(con_vals)
        pagerank = next_rank 
          
    return pagerank




if __name__ == "__main__":
    
    main()
    
