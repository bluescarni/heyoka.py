{# Template for classes without a user-facing constructor (e.g., llvm_state). #}
{# Extends the base template but excludes __init__ from both :special-members: #}
{# and the methods autosummary table. The methods block below duplicates the   #}
{# logic from custom-class-base-template.rst with an added __init__ filter -  #}
{# keep it in sync if the base template's methods block changes.               #}
{% extends "custom-class-base-template.rst" %}
{% block methods %}
   {# First pass: check if any methods besides __init__ exist, so that we   #}
   {# do not render an empty "Methods" section. We use a Jinja2 namespace   #}
   {# because plain {% set %} inside a {% for %} loop does not propagate    #}
   {# to the outer scope.                                                   #}
   {% set ns = namespace(has_methods=false) %}
   {% for item in members %}
      {% if (item in methods or item == '__call__') and item != '__init__' %}
         {% set ns.has_methods = true %}
      {% endif %}
   {%- endfor %}
   {% if ns.has_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in members %}
      {% if (item in methods or item == '__call__') and item != '__init__' %}
         ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
{% endblock %}
