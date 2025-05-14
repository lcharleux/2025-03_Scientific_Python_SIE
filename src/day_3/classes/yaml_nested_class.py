import copy
import yaml


class Container:
    def __init__(self, values=None):
        if values is None:
            values = []
        self._values = values

    def append(self, value):
        self._values.append(value)
        return self

    def copy(self):
        return self.__class__(copy.deepcopy(self._values))

    def __add__(self, value):
        return self.copy().append(value)

    def __repr__(self):
        return f"<Container({self._values})>"

    def save_to_yaml(self, filepath):
        """
        Save the container to a YAML file.
        """
        with open(filepath, "w") as file:
            yaml.dump(self, file, default_flow_style=False)

    @classmethod
    def load_from_yaml(cls, filepath):
        """
        Load a container from a YAML file.
        """
        with open(filepath, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            if isinstance(data, Container):
                return data
            raise ValueError("The YAML file does not contain a valid Container object.")

    def to_dict(self):
        """
        Convert the container to a dictionary for YAML serialization.
        """
        return {
            "values": [
                value.to_dict() if isinstance(value, Container) else value
                for value in self._values
            ]
        }

    @classmethod
    def from_dict(cls, data):
        """
        Create a container from a dictionary.
        """
        if not isinstance(data, dict) or "values" not in data:
            raise ValueError("Invalid data format for Container deserialization.")

        values = [
            cls.from_dict(value)
            if isinstance(value, dict) and "values" in value
            else value
            for value in data["values"]
        ]
        return cls(values)


# Register custom YAML serialization for the Container class
yaml.add_representer(
    Container,
    lambda dumper, data: dumper.represent_mapping("!Container", data.to_dict()),
)
yaml.add_constructor(
    "!Container",
    lambda loader, node: Container.from_dict(loader.construct_mapping(node, deep=True)),
)


# Example usage
if __name__ == "__main__":
    # Create a nested container
    c1 = Container(["hello", "world"])
    c2 = Container([c1, "Python"])
    print(c2)

    # Save to a YAML file
    c2.save_to_yaml("container.yaml")

    # Load from the YAML file
    c3 = Container.load_from_yaml("container.yaml")
    print(c3)
