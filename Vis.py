parameters = [['IM', 'OSM'], ['KommaS', 'PlusS'], ['IR', 'DR', 'GIR', 'GDR']]
for mutation_input in range(1,3):
    for selection_input in range(1,3):
        for recombination_input in range(1,5):
            for initial_sigma in range(1,11):
                for num_parents in range(2,21):
                    for num_offspring in range(2,21):
                        test = str(parameters[0][mutation_input-1]) + ' '+ str(parameters[1][selection_input-1]) + ' ' + str(parameters[2][recombination_input-1]) + ' ' + (f"IS:{initial_sigma/100}") + ' ' + (f"NP:{num_parents}") + ' ' + (f'NO:{num_offspring}')
                        print(type(test))
                        break
                    break
                break
            break
        break
    break

parameters = [['IM', 'OSM'], ['KommaS', 'PlusS'], ['IR', 'DR', 'GIR', 'GDR']]