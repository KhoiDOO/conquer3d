import random

class BaseTransform(object):
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, **data):
        if random.random() <= self.p:
            return self.apply(**data)
        return data
        
    def apply(self, **data):
        raise NotImplementedError("Transform must implement apply")

class Sequence(BaseTransform):
    def __init__(self, transforms, shuffle=False, p=1.0):
        super().__init__(p=p)
        self.transforms = transforms
        self.shuffle = shuffle

    def apply(self, **data):
        transforms_to_apply = self.transforms.copy()
        if self.shuffle:
            random.shuffle(transforms_to_apply)
            
        for transform in transforms_to_apply:
            data = transform(**data)
            
        return data

class MeshSequence(Sequence):
    def __call__(self, vertices, faces):
        if random.random() <= self.p:
            return self.apply(vertices, faces)
        return vertices, faces

    def apply(self, vertices, faces):
        data = super().apply(vertices=vertices, faces=faces)
        return data.get('vertices', vertices), data.get('faces', faces)
