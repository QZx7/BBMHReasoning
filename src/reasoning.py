def sample_generator():
    l = ["1", "2", "3", "4", "5"]
    for item in l:
        yield item


generator = sample_generator()

for sample in generator:
    print(sample)
