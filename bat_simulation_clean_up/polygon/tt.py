import os

directory = os.listdir('shape_files')


for fname in directory:
    if fname.endswith('.shp'):
        print(fname)

    # if os.path.isfile(user_input + os.sep + fname):
    #     # Full path
    #     f = open(user_input + os.sep + fname, 'r')

    #     if searchstring in f.read():
    #         print('found string in file %s' % fname)
    #     else:
    #         print('string not found')
    #     f.close()
