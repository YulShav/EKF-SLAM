"""Re-sort a rosbag by timestamp, keeping only the specified topics."""

from rosbags.rosbag1 import Reader, Writer
from pathlib import Path
import argparse


def sort_bag(input_path: Path, output_path: Path, topics: set[str] | None = None):
    with Reader(input_path) as reader:
        conn_info = {}
        messages = []
        for connection, timestamp, data in reader.messages():
            if topics and connection.topic not in topics:
                continue
            if connection.topic not in conn_info:
                conn_info[connection.topic] = (
                    connection.msgtype,
                    connection.msgdef.data,
                    connection.digest,
                )
            messages.append((connection.topic, timestamp, bytes(data)))

        print(f"Collected {len(messages)} messages, sorting...")
        messages.sort(key=lambda m: m[1])

        print("Writing sorted bag...")
        with Writer(output_path) as writer:
            conn_map = {}
            for topic, (msgtype, msgdef, digest) in conn_info.items():
                conn_map[topic] = writer.add_connection(
                    topic, msgtype, msgdef=msgdef, md5sum=digest
                )

            for topic, timestamp, data in messages:
                writer.write(conn_map[topic], timestamp, data)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Done! {output_path} ({size_mb:.1f} MB, {len(messages)} messages)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input .bag file")
    parser.add_argument("-o", "--output", type=Path, help="Output .bag file")
    parser.add_argument(
        "-t", "--topics", nargs="+", default=None,
        help="Topics to keep (default: all)",
    )
    args = parser.parse_args()

    output = args.output or args.input.with_stem(args.input.stem + "-sorted")
    sort_bag(args.input, output, set(args.topics) if args.topics else None)
