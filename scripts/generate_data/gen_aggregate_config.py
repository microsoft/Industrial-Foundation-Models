import os
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Aggregate Database Config')
    parser.add_argument('--source_dir', type=str)
    parser.add_argument('--source_database_names', type=str)
    parser.add_argument('--target_dir', type=str)
    parser.add_argument('--data_type', type=str)
    parser.add_argument('--agg_config_dir', type=str)
    parser.add_argument('--agg_config_name', type=str)
    args = parser.parse_args()
    
    info = {}
    info["source_dir"] = args.source_dir
    info["target_dir"] = args.target_dir
    info["data_type"] = args.data_type.split(',')
    
    # Load processed data
    info["source_data_list"] = []
    for database in args.source_database_names.split(','):
        for filename in os.listdir(f'{args.source_dir}/{database}'):
            if os.path.isdir(os.path.join(f'{args.source_dir}/{database}', filename)):
                info["source_data_list"].append(os.path.join(database, filename))

    # Generate aggregate config json
    info = json.dumps(info, indent=4)
    if not os.path.isdir(args.agg_config_dir):
        os.makedirs(args.agg_config_dir)
    with open(f'{args.agg_config_dir}/{args.agg_config_name}.json', 'w') as f:
        f.write(info)
