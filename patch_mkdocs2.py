import yaml
with open('mkdocs.yml', 'r') as f:
    config = yaml.safe_load(f)

for plugin in config['plugins']:
    if isinstance(plugin, dict) and 'mkdocstrings' in plugin:
        options = plugin['mkdocstrings']['handlers']['python']['options']
        options['show_if_no_docstring'] = True

with open('mkdocs.yml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
