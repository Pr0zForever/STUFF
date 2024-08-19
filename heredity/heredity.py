import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }
    print(probabilities)
    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]

def extra_mappings(people,one_gene,two_genes,have_trait):
    gene_mapping = {}
    no_genes = []
    no_trait = []
    for person in people:
        if person in one_gene:
            gene_mapping[person] = 1
        elif person in two_genes:
            gene_mapping[person] = 2
        else:
            gene_mapping[person] = 0
            no_genes.append(person)
        if person not in have_trait:
            no_trait.append(person)
    return gene_mapping,no_genes,no_trait

def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.
    """
    gene_mapping,no_genes,no_trait = extra_mappings(people,one_gene,two_genes,have_trait)
    probability = 1
    vals = {0:.01, 1:.5, 2:.99}
    
    for person in one_gene:
        if people[person]['mother']:
            mother = gene_mapping[people[person]['mother']]
            father = gene_mapping[people[person]['father']]
            probability *= (vals[mother]*(1-vals[father])) + (vals[father]*(1-vals[mother]))
        else:
            probability *= PROBS['gene'][1]
    
    for person in two_genes:
        if people[person]['mother']:
            mother = gene_mapping[people[person]['mother']]
            father = gene_mapping[people[person]['father']]
            probability *= vals[mother]* vals[father]
        else:
            probability *= PROBS['gene'][2]
    for person in no_genes:
        if people[person]['mother']:
            mother = gene_mapping[people[person]['mother']]
            father = gene_mapping[people[person]['father']]
            probability *= (1-vals[mother])* (1-vals[father])
        else:
            probability *= PROBS['gene'][0]
    for person in have_trait:
        probability *= PROBS['trait'][gene_mapping[person]][True]
    for person in no_trait:
        probability *= PROBS['trait'][gene_mapping[person]][False]
    return probability

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    people = probabilities.keys()
    gene_mapping,no_genes,no_trait = extra_mappings(people,one_gene,two_genes,have_trait)
    for person in people:
        if person in one_gene:
            probabilities[person]['gene'][1] += p
        elif person in two_genes:
            probabilities[person]['gene'][2] += p
        else:
            probabilities[person]['gene'][0] += p
        if person in have_trait:
            probabilities[person]['trait'][True] += p
        else:
            probabilities[person]['trait'][False] += p
        


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    
    for person in probabilities:
        gene_norm = sum(probabilities[person]['gene'].values())
        trait_norm = sum(probabilities[person]['trait'].values())
        for gene in probabilities[person]['gene']:
            probabilities[person]['gene'][gene] = probabilities[person]['gene'][gene]/gene_norm
        for trait in probabilities[person]['trait']:
            probabilities[person]['trait'][trait] = probabilities[person]['trait'][trait]/trait_norm
    

        


if __name__ == "__main__":
    main()
