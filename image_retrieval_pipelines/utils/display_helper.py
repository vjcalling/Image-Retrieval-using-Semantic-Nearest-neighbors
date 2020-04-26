
#display/visualization related libs
from prettytable import PrettyTable
import matplotlib.pyplot as plt

def display_results_images(result_captions):

    w=10
    h=10
    fig=plt.figure(figsize=(30, 20))
    columns = 2
    rows = 2
    for i in range(1, columns*rows +1):
        img_path = captions_df['image_files'][result_captions[i-1][3]]
        img = mpimg.imread(img_path)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis('off')
    fig.tight_layout()
    plt.show()
    #fig.savefig(IMG_DIR + str(datetime.datetime.now().timestamp()) + '.jpg')
    

def display_tabular_results(result_captions, tech='Cosine Similarity', title='Title'):
    t = PrettyTable(field_names=['({})'.format(tech), 'Caption', 'Image Path', 'Image Index'])
    t.float_format = '2.2'
    t.title = title
    
    for idx, res in enumerate(result_captions):
        t.add_row(res)
        
    print(t.get_string(title=title))
    
    #table_txt = t.get_string(title=title)
    #with open(TABLE_DIR + str(datetime.datetime.now().timestamp()),'w') as file:
    #    file.write('User query: ' + user_query + '\n')
    #    file.write('-----------')
    #    file.write('\n')
    #    file.write(table_txt)

def find_img_idx(s):
    first = 'images/'
    last = '_'
    
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return int(s[start:end])
    except ValueError:
        return ""

def display_stats(title, total, retrieved):
    print(title)
    print("------------------------------------------------\n")
    x = PrettyTable()
    x.title = title
    x.field_names = ["Total relevant images", "Relevant images retrieved", "Average Recall"]
    x.float_format['Average Recall'] = "0.2"
    x.add_row([total, retrieved, retrieved/total])
    print(x)

