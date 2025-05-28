import ast

def extract_entity_id(url):
    return url.strip().split("/")[-1]
    
def convert_str_to_list(list_string):
    try:
        list_conv = ast.literal_eval(list_string)
    except:
        list_conv = []
    return list_conv

def extract_properties(id, client):
    prop_dict = {}
    item = client.get(id, load=True)
    data = item.data['claims']
    property = list(data.keys())
    
    for prop in property:
        p_name = data[prop][0]['mainsnak']['property']
        p_values = []
        length = len(data[prop])
        for i in range(length):
            try:
                p_values.append(data[prop][i]['mainsnak']['datavalue']['value']['id'])
                prop_dict[p_name] = p_values
            except:
                ""
    return prop_dict