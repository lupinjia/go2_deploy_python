from controller import *

if __name__ == "__main__":
    import argparse

    # run ts controller in sim by default
    parser = argparse.ArgumentParser()
    parser.add_argument('--interface', '-i', type=str, default="lo", help="network interface")
    parser.add_argument("--config", type=str, default="ts.yaml", help="config file name in the configs folder")
    parser.add_argument("--type", type=str, default="ts", help="controller type: ts")
    args = parser.parse_args()

    # Load config
    config_path = f"./configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    if args.interface == "lo":
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, args.interface)
        
    if args.type == "ts":
        controller = TSController(config, args.interface)
    elif args.type == "ee":
        controller = EEController(config, args.interface)
    elif args.type == "waq":
        controller = WaQController(config, args.interface)
    else:
        raise ValueError(f"Unsupported controller type: {args.type}")
    
    while True:
        time.sleep(1)
