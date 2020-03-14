## An Adversarial Image Web Application
For a talk I put together a little while ago, I wanted a proof of concept of how easy it is to generate adversarial image examples, so I threw together a terrible web app and bolted on a simple generator. Some friends and coworkers have enjoyed it, so I'm putting up the source code so I don't have to keep paying to host it.

The app does a couple iterations of the [Fast Gradient Sign Method](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html).

Because I never really intended to share the code originally, this is not pretty code - a quick and dirty PoC thrown together over one afternoon. I don't really have any plans of maintaining or cleaning it up, but hopefully it may be useful to somebody out there.

## Requirements
You will need:
* ``uwsgi``
* ``pytorch``
* ``torchvision``
* ``Pillow``

You can start the server on port 43893 (as an example) with ``uwsgi --http :43893 --wsgi-file AdversarialServer.py``.
