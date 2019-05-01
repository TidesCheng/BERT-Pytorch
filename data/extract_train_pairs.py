import os


def extract(minlen=5, remove_list=(',', 'urlLink', '"', ';', '!', '?'),
            texts = 'bert_train_pairs.txt', line4each_file=100):
    data_dir = 'raw'

    files = os.listdir(data_dir)

    fw = open(texts, 'w')

    cnt = 0
    for file in files:
        with open(os.path.join(data_dir, file), 'r', encoding='iso-8859-15') as f:
            f_cnt = 0
            for line in f:
                f_cnt += 1
                for s in remove_list:
                    line = line.replace(s, '')
                line = line.strip()
                if line == "Link":  continue
                if line and not (line.startswith('<') or line.startswith('-')):
                    sl = line.split('.')
                    for i in range(len(sl)-1):
                        if len(sl[i].split()) > minlen and len(sl[i+1].split()) > minlen:
                            fw.write(sl[i] + ' $$$ ' + sl[i+1] + '\n')
                if f_cnt == line4each_file:
                    break
            cnt += f_cnt
    fw.close()
    print("\nTotal {} sentence pairs are extracted.".format(cnt))


if __name__=='__main__':
    import fire
    fire.Fire()