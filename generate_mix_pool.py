root_dir = '/home/tk/Documents/'

stock = []
spec0_cluster = []
spec1_cluster = []

for j in range(10):
    for i in range(100):
        from data_process import gen_spectrogram

        selected_audio = np.random.choice(g, size = 2, replace = False)

        # prevent to select from the same source
        if selected_audio[0][:8] == selected_audio[1][:8]: 
            selected_audio = np.random.choice(g, size = 2, replace = False)
        if r[0][:8] == r[1][:8]:
            selected_audio = np.random.choice(g, size = 2, replace = False)
        print (selected_audio)

        # generate 2 spectrograms & mix
        gen_path0 = '/home/tk/Documents/sliced_pool/' + selected_audio[0]
        gen_path1 = '/home/tk/Documents/sliced_pool/' + selected_audio[1]
        spec0 = gen_spectrogram(gen_path0)
        spec1 = gen_spectrogram(gen_path1)
        mixed_spec = spec0 + spec1

        # append the mixed spectrograms
        stock.append(mixed_spec)
        spec0_cluster.append(spec0)
        spec1_cluster.append(spec1)

    stock = np.array(stock)
    print ('mixed_spec shape = ', stock.shape)

    spec0_cluster = np.array(spec0_cluster)
    print ('spec0 cluster shape = ', spec0_cluster.shape)

    spec1_cluster = np.array(spec1_cluster)
    print ('spec1 cluster shape = ', spec1_cluster.shape)

    with open(root_dir + "mix_pool/mix_spec/" + str(j) + '.json', 'w') as jh:
        json.dump(stock.tolist(), jh)

    with open(root_dir + "mix_pool/spec0/" + str(j) + '.json', 'w') as jh:
        json.dump(spec0_cluster.tolist(), jh)

    with open(root_dir + "mix_pool/spec1/" + str(j) + '.json', 'w') as jh:
        json.dump(spec1_cluster.tolist(), jh)