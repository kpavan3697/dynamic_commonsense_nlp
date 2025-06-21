import csv

def load_conceptnet(filepath, lang='en'):
    triples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 4:
                continue
            uri, rel, start, end = row[0], row[1], row[2], row[3]
            if start.startswith(f'/c/{lang}/') and end.startswith(f'/c/{lang}/'):
                h = start.split('/')[3]
                r = rel.split('/')[-1]
                t = end.split('/')[3]
                triples.append((h, r, t))
    return triples